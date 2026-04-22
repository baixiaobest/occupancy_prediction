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

from src.rl import (
    ORCASimConfig,
    ORCASingleEnv,
    SingleEnvConfig,
    build_observation_manager,
    build_simple_state_observation_config,
    integrate_velocity_plans,
    sample_action_indices_from_q_scores,
    sample_random_velocity_plans,
)
from src.rl.managers.observation_manager import ObservationBatchContext
from src.scene import Scene
from src.rl.networks.simple_q_network import build_simple_q_network
from src.templates import empty_goal_templates


@dataclass
class StepRecord:
    ego_position: torch.Tensor
    goal_position: torch.Tensor
    candidate_offsets: torch.Tensor
    selected_index: int
    q_scores: torch.Tensor
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class EpisodeRecord:
    scene: Scene
    steps: list[StepRecord]
    executed_positions: torch.Tensor
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



def _build_empty_goal_scene_pool(
    *,
    goal_distance_range: tuple[float, float],
    goal_seed: int,
) -> list[Scene]:
    scenes: list[Scene] = []
    for template in empty_goal_templates(
        goal_distance_range=goal_distance_range,
        goal_seed=goal_seed,
    ):
        scenes.extend(template.generate())
    if not scenes:
        raise ValueError("No scenes generated for empty-goal template")
    return scenes



def _make_scene_factory(
    scenes: list[Scene],
    *,
    selection: str,
    fixed_scene_index: int,
    seed: int,
):
    rng = random.Random(int(seed))
    scene_count = len(scenes)
    if scene_count == 0:
        raise ValueError("scenes must not be empty")

    fixed_idx = int(fixed_scene_index) % scene_count
    next_idx = 0

    def factory() -> Scene:
        nonlocal next_idx
        if selection == "fixed":
            scene = scenes[fixed_idx]
        elif selection == "cycle":
            scene = scenes[next_idx % scene_count]
            next_idx += 1
        else:
            scene = scenes[rng.randrange(scene_count)]
        return copy.deepcopy(scene)

    return factory



def _load_simple_q_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict")

    mode = checkpoint.get("mode")
    if mode != "simple_state_q":
        raise ValueError(f"Checkpoint mode must be simple_state_q, got {mode!r}")

    q_model_config = checkpoint.get("q_model_config")
    if not isinstance(q_model_config, dict):
        raise ValueError("Checkpoint is missing q_model_config")
    if q_model_config.get("type") != "simple_state_q":
        raise ValueError("Checkpoint q_model_config.type must be simple_state_q")

    hidden_dims_raw = q_model_config.get("hidden_dims", [128, 128])
    hidden_dims = [int(v) for v in hidden_dims_raw]
    q_network = build_simple_q_network(hidden_dims=hidden_dims, device=device)

    q_state = checkpoint.get("q_network")
    if not isinstance(q_state, dict):
        raise ValueError("Checkpoint is missing q_network weights")
    q_network.load_state_dict(q_state, strict=True)
    q_network.eval()

    return q_network, checkpoint



def _extract_temperature(checkpoint: dict[str, Any], fallback: float) -> float:
    collector_config = checkpoint.get("collector_config")
    if isinstance(collector_config, dict):
        q_selection = collector_config.get("q_selection")
        if isinstance(q_selection, dict) and "temperature" in q_selection:
            value = float(q_selection["temperature"])
            if value > 0.0:
                return value
    return float(fallback)


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


def _apply_selection_to_scene(
    base_scene: Scene,
    selection: ScenarioSelection,
) -> Scene:
    scene = copy.deepcopy(base_scene)
    if not scene.agents:
        raise ValueError("Selected scene has no agents")

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


def _build_env_for_scene(
    *,
    scene: Scene,
    max_steps: int,
    device: torch.device,
) -> ORCASingleEnv:
    return ORCASingleEnv(
        scene_factory=lambda template_scene=scene: copy.deepcopy(template_scene),
        sim_config=ORCASimConfig(time_step=0.1),
        env_config=SingleEnvConfig(
            max_steps=int(max_steps),
            controlled_agent_index=0,
            device=str(device),
            observation=build_simple_state_observation_config(),
        ),
    )


def _run_selected_episode(
    *,
    q_network: torch.nn.Module,
    scenes: list[Scene],
    selection: ScenarioSelection,
    proposal_horizon: int,
    num_candidates: int,
    max_speed: float,
    delta_std: float,
    selection_temperature: float,
    env_max_steps: int,
    device: torch.device,
) -> EpisodeRecord:
    selected_scene = _apply_selection_to_scene(scenes[int(selection.scene_index)], selection)
    env = _build_env_for_scene(
        scene=selected_scene,
        max_steps=int(env_max_steps),
        device=device,
    )
    record = _run_episode(
        q_network=q_network,
        env=env,
        proposal_horizon=int(proposal_horizon),
        num_candidates=int(num_candidates),
        max_speed=float(max_speed),
        delta_std=float(delta_std),
        selection_temperature=float(selection_temperature),
        seed=int(selection.rollout_seed),
    )
    record.scenario_index = int(selection.scenario_index)
    record.scene_index = int(selection.scene_index)
    record.variant_index = int(selection.variant_index)
    record.rollout_seed = int(selection.rollout_seed)
    return record



def _run_episode(
    *,
    q_network: torch.nn.Module,
    env: ORCASingleEnv,
    proposal_horizon: int,
    num_candidates: int,
    max_speed: float,
    delta_std: float,
    selection_temperature: float,
    seed: int,
) -> EpisodeRecord:
    obs_config = build_simple_state_observation_config()
    obs_manager = build_observation_manager(obs_config)
    rng = torch.Generator(device=q_network.mlp[0].weight.device)
    rng.manual_seed(int(seed))

    raw_obs = env.reset(seed=int(seed))
    if env.sim is None:
        raise RuntimeError("Environment did not initialize simulation")
    scene = copy.deepcopy(env.sim.scene)
    controlled_idx = int(env.env_config.controlled_agent_index)

    obs = obs_manager.compute(ObservationBatchContext(raw_obs=raw_obs, scene=env.sim.scene))

    executed_positions: list[torch.Tensor] = [
        torch.as_tensor(raw_obs["positions"], dtype=torch.float32)[0, controlled_idx].detach().cpu()
    ]
    steps: list[StepRecord] = []

    max_steps = int(env.env_config.max_steps)
    dt = float(env.sim_config.time_step)

    for _ in range(max_steps):
        current_position = torch.as_tensor(raw_obs["positions"], dtype=torch.float32)[0, controlled_idx]
        goal_position = torch.as_tensor(raw_obs["goals"], dtype=torch.float32)[0, controlled_idx]

        candidate_plans = sample_random_velocity_plans(
            current_velocity=obs["current_velocity"],
            num_candidates=int(num_candidates),
            horizon=int(proposal_horizon),
            max_speed=float(max_speed),
            delta_std=float(delta_std),
            dt=dt,
            include_current_velocity_candidate=True,
            generator=rng,
        )

        candidate_actions = candidate_plans[:, :, 0, :]
        batch_size, k = candidate_actions.shape[:2]
        current_velocity = obs["current_velocity"][:, None, :].expand(-1, k, -1).reshape(batch_size * k, -1)
        goal_offset = obs["goal_position"][:, None, :].expand(-1, k, -1).reshape(batch_size * k, -1)
        flat_actions = candidate_actions.reshape(batch_size * k, -1)

        with torch.no_grad():
            q_scores = q_network(
                current_velocity=current_velocity,
                goal_position=goal_offset,
                action=flat_actions,
            ).reshape(batch_size, k)

        selected_indices, _ = sample_action_indices_from_q_scores(
            q_scores,
            temperature=float(selection_temperature),
            generator=rng,
        )
        selected_idx = int(selected_indices[0].item())

        candidate_offsets = integrate_velocity_plans(candidate_plans, dt=dt)[0].detach().cpu()
        selected_action = candidate_actions[0, selected_idx]

        next_raw_obs, rewards, dones, infos = env.step(selected_action)

        reward = float(torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)[0].item())
        done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
        info = dict(infos[0])

        steps.append(
            StepRecord(
                ego_position=current_position.detach().cpu(),
                goal_position=goal_position.detach().cpu(),
                candidate_offsets=candidate_offsets,
                selected_index=selected_idx,
                q_scores=q_scores[0].detach().cpu(),
                reward=reward,
                done=done,
                info=info,
            )
        )

        next_position = torch.as_tensor(next_raw_obs["positions"], dtype=torch.float32)[0, controlled_idx]
        executed_positions.append(next_position.detach().cpu())

        raw_obs = next_raw_obs
        obs = obs_manager.compute(ObservationBatchContext(raw_obs=raw_obs, scene=env.sim.scene))
        if done:
            break

    return EpisodeRecord(
        scene=scene,
        steps=steps,
        executed_positions=torch.stack(executed_positions, dim=0),
    )



def _compute_plot_limits(record: EpisodeRecord) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    for obstacle in record.scene.obstacles:
        for x, y in obstacle.vertices:
            xs.append(float(x))
            ys.append(float(y))

    for point in record.executed_positions:
        xs.append(float(point[0].item()))
        ys.append(float(point[1].item()))

    for step in record.steps:
        current = step.ego_position.numpy()
        xs.append(float(current[0]))
        ys.append(float(current[1]))
        goal = step.goal_position.numpy()
        xs.append(float(goal[0]))
        ys.append(float(goal[1]))

        candidate_endpoints = step.candidate_offsets[:, -1, :].numpy() + current[None, :]
        xs.extend(float(v) for v in candidate_endpoints[:, 0])
        ys.extend(float(v) for v in candidate_endpoints[:, 1])

    if not xs or not ys:
        return -5.0, 5.0, -5.0, 5.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(1.0, 0.15 * (max_x - min_x + 1e-6))
    pad_y = max(1.0, 0.15 * (max_y - min_y + 1e-6))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y



def _render(record: EpisodeRecord, step_idx: int, ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    ax.clear()
    min_x, max_x, min_y, max_y = limits

    step = record.steps[step_idx]
    current = step.ego_position.numpy()
    goal = step.goal_position.numpy()
    candidate_offsets = step.candidate_offsets.numpy()

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

    history = record.executed_positions[: step_idx + 1].numpy()
    ax.plot(history[:, 0], history[:, 1], color="tab:blue", linewidth=2.4, label="executed trajectory", zorder=4)
    ax.scatter(history[-1, 0], history[-1, 1], color="tab:blue", s=40, zorder=5)

    k = candidate_offsets.shape[0]
    colors = plt.cm.Greys(np.linspace(0.45, 0.9, max(k, 2)))
    selected_idx = int(step.selected_index)

    for idx in range(k):
        traj = np.concatenate(
            [
                current[None, :],
                current[None, :] + candidate_offsets[idx],
            ],
            axis=0,
        )
        if idx == selected_idx:
            continue
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=colors[idx % len(colors)],
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )

    selected_traj = np.concatenate(
        [
            current[None, :],
            current[None, :] + candidate_offsets[selected_idx],
        ],
        axis=0,
    )
    ax.plot(
        selected_traj[:, 0],
        selected_traj[:, 1],
        color="tab:red",
        linewidth=2.8,
        alpha=0.95,
        label="selected trajectory",
        zorder=6,
    )
    ax.scatter(selected_traj[-1, 0], selected_traj[-1, 1], color="tab:red", s=36, zorder=7)

    ax.scatter(goal[0], goal[1], marker="*", s=240, color="gold", edgecolor="black", linewidth=0.7, label="goal", zorder=8)

    q_scores = step.q_scores.numpy()
    q_min = float(np.min(q_scores))
    q_max = float(np.max(q_scores))
    q_sel = float(q_scores[selected_idx])

    info = step.info
    goal_dist = float(info.get("goal_distance", float("nan")))
    success = bool(info.get("success", False))
    collision = bool(info.get("collision", False))
    timeout = bool(info.get("timeout", False))

    ax.set_title(
        f"Simple-Q rollout | setup={record.scenario_index + 1} (scene={record.scene_index}, variant={record.variant_index}) | "
        f"step {step_idx + 1}/{len(record.steps)} | selected={selected_idx} | "
        f"reward={step.reward:.3f} | goal_dist={goal_dist:.3f}\n"
        f"Q range=[{q_min:.3f}, {q_max:.3f}] | Q(selected)={q_sel:.3f} | "
        f"done={step.done} (success={success}, collision={collision}, timeout={timeout})"
    )

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=2.4, label="executed trajectory"),
        Line2D([0], [0], color="0.65", lw=1.5, label="proposed trajectories"),
        Line2D([0], [0], color="tab:red", lw=2.8, label="selected trajectory"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=12, label="goal"),
    ]
    ax.legend(handles=legend_handles, loc="best")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize simple-Q policy rollout with proposed and selected trajectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rl_q_selection_debug_iter_001000.pt"),
        help="Path to simple_state_q checkpoint",
    )
    parser.add_argument(
        "--scene-selection",
        choices=["random", "cycle", "fixed"],
        default="fixed",
        help="How to pick scenes from generated empty-goal scene pool",
    )
    parser.add_argument("--fixed-scene-index", type=int, default=0)
    parser.add_argument("--goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--goal-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-max-steps", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--proposal-horizon", type=int, default=12)
    parser.add_argument("--scenario-variants-per-scene", type=int, default=4)
    parser.add_argument("--candidate-max-speed", type=float, default=2.0)
    parser.add_argument("--candidate-delta-std", type=float, default=1.0)
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for candidate selection; defaults to checkpoint value when available",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if int(args.num_candidates) <= 0:
        raise ValueError("--num-candidates must be > 0")
    if int(args.proposal_horizon) <= 0:
        raise ValueError("--proposal-horizon must be > 0")
    if float(args.candidate_max_speed) <= 0.0:
        raise ValueError("--candidate-max-speed must be > 0")
    if float(args.candidate_delta_std) < 0.0:
        raise ValueError("--candidate-delta-std must be >= 0")
    if int(args.env_max_steps) <= 0:
        raise ValueError("--env-max-steps must be > 0")
    if int(args.scenario_variants_per_scene) <= 0:
        raise ValueError("--scenario-variants-per-scene must be > 0")

    device = torch.device(args.device)
    q_network, checkpoint = _load_simple_q_checkpoint(args.checkpoint, device)

    goal_range = (float(args.goal_distance_range[0]), float(args.goal_distance_range[1]))
    scenes = _build_empty_goal_scene_pool(
        goal_distance_range=goal_range,
        goal_seed=int(args.goal_seed),
    )
    variants_per_scene = int(args.scenario_variants_per_scene)
    total_setups = len(scenes) * variants_per_scene

    initial_setup_index = int(args.fixed_scene_index) * variants_per_scene
    selection = _build_scenario_selection(
        scenario_index=initial_setup_index,
        scene_count=len(scenes),
        variants_per_scene=variants_per_scene,
        base_seed=int(args.seed),
    )

    selection_temperature = _extract_temperature(checkpoint, fallback=float(args.selection_temperature))

    record = _run_selected_episode(
        q_network=q_network,
        scenes=scenes,
        selection=selection,
        proposal_horizon=int(args.proposal_horizon),
        num_candidates=int(args.num_candidates),
        max_speed=float(args.candidate_max_speed),
        delta_std=float(args.candidate_delta_std),
        selection_temperature=selection_temperature,
        env_max_steps=int(args.env_max_steps),
        device=device,
    )

    if not record.steps:
        raise RuntimeError("No simulation steps were recorded")

    limits = _compute_plot_limits(record)
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
        start_xy = active_record.executed_positions[0].detach().cpu().numpy()
        goal_xy = active_record.steps[0].goal_position.detach().cpu().numpy()
        setup_label.set_text(
            f"Setup {active_record.scenario_index + 1}/{total_setups} | "
            f"scene={active_record.scene_index}, variant={active_record.variant_index}, rollout_seed={active_record.rollout_seed} | "
            f"start=({start_xy[0]:.2f}, {start_xy[1]:.2f}) goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )

    def _render_step(step_idx: int) -> None:
        active_record: EpisodeRecord = state["record"]
        active_limits: tuple[float, float, float, float] = state["limits"]
        _render(active_record, int(step_idx), ax, active_limits)
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
            q_network=q_network,
            scenes=scenes,
            selection=selection_local,
            proposal_horizon=int(args.proposal_horizon),
            num_candidates=int(args.num_candidates),
            max_speed=float(args.candidate_max_speed),
            delta_std=float(args.candidate_delta_std),
            selection_temperature=selection_temperature,
            env_max_steps=int(args.env_max_steps),
            device=device,
        )
        if not record_local.steps:
            raise RuntimeError("Selected setup produced an empty rollout")

        state["selection"] = selection_local
        state["record"] = record_local
        state["limits"] = _compute_plot_limits(record_local)

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
