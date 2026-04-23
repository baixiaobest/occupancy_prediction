from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scene import Scene
from src.templates import (
    cross_templates,
    default_templates,
    empty_goal_templates,
    l_shape_templates,
    test_templates,
)
from sb3.env_orca import ORCASB3Env, ORCASB3EnvConfig, ORCASB3RewardConfig, ORCASB3SimConfig

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from exc


@dataclass
class StepRecord:
    ego_position: np.ndarray
    current_velocity: np.ndarray
    goal_position: np.ndarray
    all_positions: np.ndarray
    commanded_velocity: np.ndarray
    value: float
    log_prob: float
    entropy: float
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeRecord:
    scene: Scene
    steps: list[StepRecord]
    executed_positions: np.ndarray
    controlled_agent_index: int = 0
    scenario_index: int = 0
    scene_index: int = 0
    variant_index: int = 0
    rollout_seed: int = 0


@dataclass
class ScenarioSelection:
    scenario_index: int
    scene_index: int
    variant_index: int
    rollout_seed: int
    start_shift: tuple[float, float]
    goal_shift: tuple[float, float]


def _scene_has_controllable_agent(scene: Scene) -> bool:
    if len(scene.agents) > 0:
        return True
    if not scene.region_pairs:
        return False

    startup_count = sum(max(0, int(getattr(pair, "startup_agent_count", 0))) for pair in scene.region_pairs)
    return startup_count > 0


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path

    if path.suffix != ".zip":
        zip_path = path.with_suffix(path.suffix + ".zip") if path.suffix else path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path

    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _select_templates(template_set: str):
    if template_set == "default":
        return default_templates()
    if template_set == "test":
        return test_templates()
    if template_set == "cross":
        return cross_templates()
    if template_set == "l_shape":
        return l_shape_templates()
    if template_set == "empty_goal":
        return empty_goal_templates()
    raise ValueError(f"Unknown template set: {template_set}")


def _build_scene_pool(
    *,
    template_set: str,
    goal_distance_range: tuple[float, float],
    goal_seed: int,
    empty_goal_other_agents_range: tuple[int, int],
    empty_goal_other_spawn_radius_range: tuple[float, float],
    empty_goal_other_goal_distance_range: tuple[float, float],
    empty_goal_other_min_start_separation: float,
) -> list[Scene]:
    if template_set == "empty_goal":
        templates = empty_goal_templates(
            goal_distance_range=goal_distance_range,
            goal_seed=goal_seed,
            num_other_agents_range=empty_goal_other_agents_range,
            other_agent_spawn_radius_range=empty_goal_other_spawn_radius_range,
            other_agent_goal_distance_range=empty_goal_other_goal_distance_range,
            other_agent_min_start_separation=float(empty_goal_other_min_start_separation),
        )
    else:
        templates = _select_templates(template_set)

    scenes: list[Scene] = []
    for template in templates:
        scenes.extend(template.generate())

    if not scenes:
        raise ValueError(f"No scenes generated for template set: {template_set}")

    valid_scenes = [scene for scene in scenes if _scene_has_controllable_agent(scene)]
    if not valid_scenes:
        raise ValueError(
            "Generated scenes contain no controllable agents. "
            "Need either explicit scene.agents or region_pairs with startup_agent_count > 0."
        )
    return valid_scenes


def _build_scenario_selection(
    *,
    scenario_index: int,
    scene_count: int,
    variants_per_scene: int,
    base_seed: int,
) -> ScenarioSelection:
    if scene_count <= 0:
        raise ValueError("scene_count must be > 0")

    per_scene = max(1, int(variants_per_scene))
    total = int(scene_count) * per_scene
    slot = int(scenario_index) % total
    scene_index = slot // per_scene
    variant_index = slot % per_scene

    rng = random.Random(int(base_seed) + 10007 * scene_index + 7919 * variant_index)
    start_shift = (float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0)))
    goal_shift = (float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-2.5, 2.5)))
    rollout_seed = int(base_seed) + 97 * slot

    return ScenarioSelection(
        scenario_index=slot,
        scene_index=scene_index,
        variant_index=variant_index,
        rollout_seed=rollout_seed,
        start_shift=start_shift,
        goal_shift=goal_shift,
    )


def _apply_selection_to_scene(base_scene: Scene, selection: ScenarioSelection) -> Scene:
    scene = copy.deepcopy(base_scene)
    if not scene.agents:
        # Region-pair scenes are allowed to start empty; ORCASim spawns startup agents on reset.
        return scene

    controlled_agent = scene.agents[0]
    start_x = float(controlled_agent.position[0]) + float(selection.start_shift[0])
    start_y = float(controlled_agent.position[1]) + float(selection.start_shift[1])
    goal_x = float(controlled_agent.goal[0]) + float(selection.goal_shift[0])
    goal_y = float(controlled_agent.goal[1]) + float(selection.goal_shift[1])

    dx = goal_x - start_x
    dy = goal_y - start_y
    goal_dist = float(np.hypot(dx, dy))
    if goal_dist < 1.0:
        goal_x = start_x + 1.0
        goal_y = start_y

    controlled_agent.position = (start_x, start_y)
    controlled_agent.goal = (goal_x, goal_y)

    if scene.ego_centers:
        centers = list(scene.ego_centers)
        centers[0] = (start_x, start_y)
        scene.ego_centers = centers
    else:
        scene.ego_centers = [(start_x, start_y)]

    return scene


def _build_env_for_scene(*, scene: Scene, config: ORCASB3EnvConfig) -> ORCASB3Env:
    return ORCASB3Env(
        scene_factory=lambda template_scene=scene: copy.deepcopy(template_scene),
        config=copy.deepcopy(config),
    )


def _load_ppo_model(checkpoint_path: Path, device: str) -> PPO:
    resolved = _resolve_checkpoint_path(checkpoint_path)
    model = PPO.load(
        str(resolved),
        device=device,
    )
    model.policy.set_training_mode(False)
    return model


def _run_selected_episode(
    *,
    model: PPO,
    scenes: list[Scene],
    selection: ScenarioSelection,
    env_config: ORCASB3EnvConfig,
    deterministic: bool,
) -> EpisodeRecord:
    scene = _apply_selection_to_scene(scenes[int(selection.scene_index)], selection)
    env = _build_env_for_scene(scene=scene, config=env_config)

    obs, _ = env.reset(seed=int(selection.rollout_seed))
    if env.sim is None or env._goals is None or env._last_positions is None or env._last_velocities is None:
        raise RuntimeError("Environment failed to initialize simulator state")

    controlled_idx = int(env.config.controlled_agent_index)
    executed_positions: list[np.ndarray] = [env._last_positions[controlled_idx].copy()]
    steps: list[StepRecord] = []

    for _ in range(int(env.config.max_steps)):
        if env._last_positions is None or env._goals is None or env._last_velocities is None:
            raise RuntimeError("Environment state unexpectedly missing during rollout")

        all_positions = env._last_positions.copy()
        current_pos = env._last_positions[controlled_idx].copy()
        current_vel = env._last_velocities[controlled_idx].copy()
        goal_pos = env._goals[controlled_idx].copy()

        obs_tensor, _ = model.policy.obs_to_tensor(obs)

        action_output, _ = model.predict(obs, deterministic=bool(deterministic))
        action_output = np.asarray(action_output, dtype=np.float32).reshape(2)

        action_output_tensor = torch.as_tensor(
            action_output,
            dtype=torch.float32,
            device=model.device,
        ).unsqueeze(0)
        with torch.no_grad():
            values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, action_output_tensor)

        next_obs, reward, terminated, truncated, info = env.step(action_output)
        commanded_velocity = np.asarray(next_obs["last_commanded_velocity"], dtype=np.float32).reshape(2)

        steps.append(
            StepRecord(
                ego_position=current_pos,
                current_velocity=current_vel,
                goal_position=goal_pos,
                all_positions=all_positions,
                commanded_velocity=commanded_velocity.copy(),
                value=float(values[0].item()),
                log_prob=float(log_prob[0].item()),
                entropy=float(entropy[0].item()),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=dict(info),
            )
        )

        if env._last_positions is None:
            raise RuntimeError("Environment positions missing after step")
        executed_positions.append(env._last_positions[controlled_idx].copy())

        obs = next_obs
        if bool(terminated) or bool(truncated):
            break

    return EpisodeRecord(
        scene=copy.deepcopy(scene),
        steps=steps,
        executed_positions=np.stack(executed_positions, axis=0),
        controlled_agent_index=controlled_idx,
        scenario_index=int(selection.scenario_index),
        scene_index=int(selection.scene_index),
        variant_index=int(selection.variant_index),
        rollout_seed=int(selection.rollout_seed),
    )


def _compute_plot_limits(record: EpisodeRecord, dt: float) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    for obstacle in record.scene.obstacles:
        for x, y in obstacle.vertices:
            xs.append(float(x))
            ys.append(float(y))

    for point in record.executed_positions:
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    for step in record.steps:
        for point in step.all_positions:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

        current = step.ego_position
        goal = step.goal_position
        xs.extend([float(current[0]), float(goal[0])])
        ys.extend([float(current[1]), float(goal[1])])

        current_vel_end = current + dt * step.current_velocity
        xs.append(float(current_vel_end[0]))
        ys.append(float(current_vel_end[1]))

        commanded_vel_end = current + dt * step.commanded_velocity
        xs.append(float(commanded_vel_end[0]))
        ys.append(float(commanded_vel_end[1]))

    if not xs or not ys:
        return -5.0, 5.0, -5.0, 5.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(1.0, 0.15 * (max_x - min_x + 1e-6))
    pad_y = max(1.0, 0.15 * (max_y - min_y + 1e-6))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _render(
    record: EpisodeRecord,
    step_idx: int,
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    dt: float,
    deterministic: bool,
) -> None:
    ax.clear()
    min_x, max_x, min_y, max_y = limits
    velocity_draw_scale = 0.5

    step = record.steps[step_idx]
    current = step.ego_position
    goal = step.goal_position

    for obstacle in record.scene.obstacles:
        polygon = Polygon(
            obstacle.vertices,
            closed=True,
            facecolor="0.87",
            edgecolor="0.45",
            linewidth=1.0,
            zorder=0,
        )
        ax.add_patch(polygon)

    history = record.executed_positions[: step_idx + 1]
    ax.plot(history[:, 0], history[:, 1], color="tab:blue", linewidth=2.4, label="executed trajectory", zorder=4)

    controlled_idx = int(record.controlled_agent_index)
    if step.all_positions.shape[0] > 1:
        other_mask = np.arange(step.all_positions.shape[0]) != controlled_idx
        other_positions = step.all_positions[other_mask]
        if other_positions.size > 0:
            ax.scatter(
                other_positions[:, 0],
                other_positions[:, 1],
                color="tab:gray",
                s=28,
                alpha=0.85,
                label="other agents",
                zorder=4,
            )

    ax.scatter(current[0], current[1], color="tab:blue", s=40, zorder=5)

    ax.quiver(
        current[0],
        current[1],
        velocity_draw_scale * step.current_velocity[0],
        velocity_draw_scale * step.current_velocity[1],
        color="tab:green",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        alpha=0.9,
        zorder=6,
    )
    ax.quiver(
        current[0],
        current[1],
        velocity_draw_scale * step.commanded_velocity[0],
        velocity_draw_scale * step.commanded_velocity[1],
        color="tab:red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        alpha=0.9,
        zorder=7,
    )

    ax.scatter(goal[0], goal[1], marker="*", s=240, color="gold", edgecolor="black", linewidth=0.7, label="goal", zorder=8)

    info = step.info
    goal_dist = float(info.get("goal_distance", float("nan")))
    success = bool(info.get("success", False))
    collision = bool(info.get("collision", False))
    timeout = bool(info.get("timeout", False))

    current_vel_text = np.array2string(step.current_velocity, precision=2, suppress_small=True)
    commanded_vel_text = np.array2string(step.commanded_velocity, precision=2, suppress_small=True)
    action_mode = "mean(deterministic)" if deterministic else "single_sample(stochastic)"

    ax.set_title(
        f"SB3 PPO rollout | setup={record.scenario_index + 1} (scene={record.scene_index}, variant={record.variant_index}) | "
        f"step {step_idx + 1}/{len(record.steps)}\n"
        f"reward={step.reward:.3f} | value={step.value:.3f} | log_prob={step.log_prob:.3f} | entropy={step.entropy:.3f} | "
        f"goal_dist={goal_dist:.3f} | done={step.terminated or step.truncated} "
        f"(success={success}, collision={collision}, timeout={timeout})\n"
        f"policy={action_mode} | current_vel={current_vel_text} | commanded_vel={commanded_vel_text}"
    )

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=2.4, label="executed trajectory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:gray", markersize=7, label="other agents"),
        Line2D([0], [0], color="tab:green", lw=2.0, label="current velocity"),
        Line2D([0], [0], color="tab:red", lw=2.0, label="commanded velocity"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=12, label="goal"),
    ]
    ax.legend(handles=legend_handles, loc="best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SB3 PPO policy rollout in ORCA scenes")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sb3_ppo_orca.zip"),
        help="Path to SB3 PPO checkpoint (.zip). If omitted suffix, .zip is tried.",
    )
    parser.add_argument(
        "--template-set",
        choices=["default", "test", "cross", "l_shape", "empty_goal"],
        default="default",
    )
    parser.add_argument("--goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--goal-seed", type=int, default=42)
    parser.add_argument("--empty-goal-other-agents-range", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--empty-goal-other-spawn-radius-range", type=float, nargs=2, default=[1.5, 6.0])
    parser.add_argument("--empty-goal-other-goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--empty-goal-other-min-start-separation", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-variants-per-scene", type=int, default=4)

    parser.add_argument("--env-max-steps", type=int, default=120)
    parser.add_argument("--controlled-agent-index", type=int, default=0)
    parser.add_argument("--max-speed", type=float, default=3.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.2)

    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--collision-penalty", type=float, default=-1.0)
    parser.add_argument("--success-reward", type=float, default=5.0)
    parser.add_argument("--collision-distance", type=float, default=0.4)

    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.scenario_variants_per_scene) <= 0:
        raise ValueError("--scenario-variants-per-scene must be > 0")
    if int(args.env_max_steps) <= 0:
        raise ValueError("--env-max-steps must be > 0")
    other_agents_range = (
        int(args.empty_goal_other_agents_range[0]),
        int(args.empty_goal_other_agents_range[1]),
    )
    if other_agents_range[0] < 0 or other_agents_range[1] < 0:
        raise ValueError("--empty-goal-other-agents-range values must be >= 0")
    if float(args.empty_goal_other_min_start_separation) < 0.0:
        raise ValueError("--empty-goal-other-min-start-separation must be >= 0")

    model = _load_ppo_model(args.checkpoint, device=str(args.device))

    goal_range = (float(args.goal_distance_range[0]), float(args.goal_distance_range[1]))
    other_spawn_radius_range = (
        float(args.empty_goal_other_spawn_radius_range[0]),
        float(args.empty_goal_other_spawn_radius_range[1]),
    )
    other_goal_distance_range = (
        float(args.empty_goal_other_goal_distance_range[0]),
        float(args.empty_goal_other_goal_distance_range[1]),
    )
    scenes = _build_scene_pool(
        template_set=str(args.template_set),
        goal_distance_range=goal_range,
        goal_seed=int(args.goal_seed),
        empty_goal_other_agents_range=other_agents_range,
        empty_goal_other_spawn_radius_range=other_spawn_radius_range,
        empty_goal_other_goal_distance_range=other_goal_distance_range,
        empty_goal_other_min_start_separation=float(args.empty_goal_other_min_start_separation),
    )

    env_config = ORCASB3EnvConfig(
        max_steps=int(args.env_max_steps),
        controlled_agent_index=int(args.controlled_agent_index),
        sim=ORCASB3SimConfig(
            max_speed=float(args.max_speed),
            goal_tolerance=float(args.goal_tolerance),
        ),
        reward=ORCASB3RewardConfig(
            progress_weight=float(args.progress_weight),
            step_penalty=float(args.step_penalty),
            collision_penalty=float(args.collision_penalty),
            success_reward=float(args.success_reward),
            collision_distance=float(args.collision_distance),
        ),
    )

    variants_per_scene = int(args.scenario_variants_per_scene)
    total_setups = len(scenes) * variants_per_scene

    initial_setup_index = 0
    selection = _build_scenario_selection(
        scenario_index=initial_setup_index,
        scene_count=len(scenes),
        variants_per_scene=variants_per_scene,
        base_seed=int(args.seed),
    )

    record = _run_selected_episode(
        model=model,
        scenes=scenes,
        selection=selection,
        env_config=env_config,
        deterministic=bool(args.deterministic),
    )

    if not record.steps:
        raise RuntimeError("No simulation steps were recorded")

    dt = float(env_config.sim.time_step)
    limits = _compute_plot_limits(record, dt=dt)
    state: dict[str, Any] = {
        "selection": selection,
        "record": record,
        "limits": limits,
    }

    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

    setup_prev_ax = fig.add_axes([0.08, 0.18, 0.12, 0.04])
    setup_next_ax = fig.add_axes([0.82, 0.18, 0.12, 0.04])
    setup_label = fig.text(0.22, 0.235, "", fontsize=10)

    slider_ax = fig.add_axes([0.18, 0.10, 0.62, 0.04])
    prev_ax = fig.add_axes([0.08, 0.10, 0.08, 0.04])
    next_ax = fig.add_axes([0.82, 0.10, 0.08, 0.04])

    step_slider = Slider(
        slider_ax,
        "Step",
        0,
        len(record.steps) - 1,
        valinit=0,
        valstep=1,
    )
    prev_button = Button(prev_ax, "Prev step")
    next_button = Button(next_ax, "Next step")
    setup_prev_button = Button(setup_prev_ax, "Prev setup")
    setup_next_button = Button(setup_next_ax, "Next setup")

    callbacks_suppressed = {"active": False}

    def _update_slider_range(slider: Slider, new_max: int) -> None:
        slider.valmax = float(new_max)
        slider.ax.set_xlim(slider.valmin, float(new_max))

    def _update_setup_label() -> None:
        active_record: EpisodeRecord = state["record"]
        start_xy = active_record.executed_positions[0]
        goal_xy = active_record.steps[0].goal_position
        setup_label.set_text(
            f"Setup {active_record.scenario_index + 1}/{total_setups} | "
            f"scene={active_record.scene_index}, variant={active_record.variant_index}, rollout_seed={active_record.rollout_seed} | "
            f"start=({start_xy[0]:.2f}, {start_xy[1]:.2f}) goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )

    def _render_step(step_idx: int) -> None:
        active_record: EpisodeRecord = state["record"]
        active_limits: tuple[float, float, float, float] = state["limits"]
        _render(
            active_record,
            int(step_idx),
            ax,
            active_limits,
            dt=dt,
            deterministic=bool(args.deterministic),
        )
        _update_setup_label()
        fig.canvas.draw_idle()

    def _load_setup(setup_index: int) -> None:
        selection_local = _build_scenario_selection(
            scenario_index=int(setup_index),
            scene_count=len(scenes),
            variants_per_scene=variants_per_scene,
            base_seed=int(args.seed),
        )
        record_local = _run_selected_episode(
            model=model,
            scenes=scenes,
            selection=selection_local,
            env_config=env_config,
            deterministic=bool(args.deterministic),
        )
        if not record_local.steps:
            raise RuntimeError("Selected setup produced an empty rollout")

        state["selection"] = selection_local
        state["record"] = record_local
        state["limits"] = _compute_plot_limits(record_local, dt=dt)

        callbacks_suppressed["active"] = True
        _update_slider_range(step_slider, len(record_local.steps) - 1)
        if int(step_slider.val) != 0:
            step_slider.set_val(0)
        callbacks_suppressed["active"] = False
        _render_step(0)

    def on_step_change(value: float) -> None:
        if callbacks_suppressed["active"]:
            return
        _render_step(int(value))

    def move_step(delta: int) -> None:
        current = int(step_slider.val)
        active_record: EpisodeRecord = state["record"]
        nxt = max(0, min(len(active_record.steps) - 1, current + delta))
        if nxt != current:
            step_slider.set_val(nxt)

    def move_setup(delta: int) -> None:
        active_selection: ScenarioSelection = state["selection"]
        target_setup = int(active_selection.scenario_index) + int(delta)
        _load_setup(target_setup)

    prev_button.on_clicked(lambda _event: move_step(-1))
    next_button.on_clicked(lambda _event: move_step(1))
    setup_prev_button.on_clicked(lambda _event: move_setup(-1))
    setup_next_button.on_clicked(lambda _event: move_setup(1))
    step_slider.on_changed(on_step_change)

    _render_step(0)
    plt.show()


if __name__ == "__main__":
    main()
