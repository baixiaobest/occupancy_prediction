from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skrl.envs.wrappers.torch import wrap_env

from src.sb3.utils import load_decoder_context_len_from_checkpoint
from src.skrl.env_torch_orca import (
    TorchORCAEnv,
    TorchORCAEnvConfig,
    TorchORCARewardConfig,
    TorchORCASimConfig,
)
from src.skrl.models import OccupancyPolicyModel, OccupancyQValueModel, OccupancyValueModel
from src.skrl.observation_wrappers import MinimalKinematicsObservationWrapper

# Reuse scene sampling and plotting utilities from SB3 visualizer.
from visualize_sb3_rollout import (
    EpisodeRecord,
    ScenarioSelection,
    StepRecord,
    _apply_selection_to_scene,
    _build_scene_pool,
    _build_scenario_selection,
    _compute_plot_limits,
    _render,
)


FIXED_GRID_SIZE_METERS = 15.0


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _load_checkpoint_dict(*, checkpoint_path: Path, device: torch.device) -> dict:
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    try:
        checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(resolved_checkpoint, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("SKRL checkpoint must be a dict with at least a 'policy' state_dict")

    _select_policy_and_aux_state_dicts(checkpoint)

    return checkpoint


def _select_policy_and_aux_state_dicts(
    checkpoint: Mapping[str, object],
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], str]:
    policy_state_dict = checkpoint.get("policy")
    if not isinstance(policy_state_dict, Mapping):
        raise ValueError("SKRL checkpoint missing required key: 'policy'")

    value_state_dict = checkpoint.get("value")
    if isinstance(value_state_dict, Mapping):
        return policy_state_dict, value_state_dict, "ppo"

    critic_state_dict = checkpoint.get("critic_1")
    if isinstance(critic_state_dict, Mapping):
        return policy_state_dict, critic_state_dict, "sac"

    raise ValueError(
        "SKRL checkpoint missing required critic/value key: expected 'value' (PPO) or 'critic_1' (SAC)"
    )


def _checkpoint_uses_legacy_feature_extractor(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(str(name).startswith("feature_extractor.") for name in state_dict.keys())


def _checkpoint_uses_tap_projector(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(str(name).startswith("_tap_projector.tap_bottleneck.") for name in state_dict.keys())


def _infer_tap_bottleneck_arch(state_dict: Mapping[str, torch.Tensor]) -> tuple[tuple[int, ...], int]:
    pattern = re.compile(r"^_tap_projector\.tap_bottleneck\.(\d+)\.weight$")
    linear_layers: list[tuple[int, int]] = []
    for key, value in state_dict.items():
        match = pattern.match(str(key))
        if match is None:
            continue
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        layer_index = int(match.group(1))
        out_features = int(value.shape[0])
        linear_layers.append((layer_index, out_features))

    if not linear_layers:
        raise ValueError("Unable to infer tap bottleneck architecture from checkpoint")

    linear_layers.sort(key=lambda item: item[0])
    out_dims = [dim for _, dim in linear_layers]
    hidden_dims = tuple(int(v) for v in out_dims[:-1])
    output_dim = int(out_dims[-1])
    return hidden_dims, output_dim


def _infer_checkpoint_map_extractor_type(checkpoint: dict) -> str:
    policy_state_dict, aux_state_dict, _ = _select_policy_and_aux_state_dicts(checkpoint)

    policy_uses_legacy_extractor = _checkpoint_uses_legacy_feature_extractor(policy_state_dict)
    aux_uses_legacy_extractor = _checkpoint_uses_legacy_feature_extractor(aux_state_dict)
    if policy_uses_legacy_extractor or aux_uses_legacy_extractor:
        raise ValueError(
            "This checkpoint uses the legacy feature_extractor path. "
            "Use a checkpoint trained with env-side decoder tapping."
        )

    policy_uses_tap_projector = _checkpoint_uses_tap_projector(policy_state_dict)
    aux_uses_tap_projector = _checkpoint_uses_tap_projector(aux_state_dict)
    if policy_uses_tap_projector != aux_uses_tap_projector:
        raise ValueError(
            "Checkpoint appears inconsistent: policy and auxiliary critic/value disagree on tap bottleneck weights"
        )
    return "vae_tap" if policy_uses_tap_projector else "conv"


def _build_env_config(
    args: argparse.Namespace,
    *,
    effective_map_extractor_type: str | None = None,
) -> TorchORCAEnvConfig:
    env_config = TorchORCAEnvConfig(
        max_steps=int(args.env_max_steps),
        controlled_agent_index=int(args.controlled_agent_index),
        device=str(args.device),
        sim=TorchORCASimConfig(
            max_speed=float(args.max_speed),
            goal_tolerance=float(args.goal_tolerance),
        ),
        reward=TorchORCARewardConfig(
            progress_weight=float(args.progress_weight),
            step_penalty=float(args.step_penalty),
            collision_penalty=float(args.collision_penalty),
            success_reward=float(args.success_reward),
            collision_distance=float(args.collision_distance),
        ),
    )

    extractor_type = str(
        args.map_extractor_type if effective_map_extractor_type is None else effective_map_extractor_type
    ).strip().lower()
    env_config.map_extractor_type = extractor_type
    if extractor_type == "vae_tap":
        if args.vae_checkpoint is None:
            raise ValueError("--vae-checkpoint is required when --map-extractor-type=vae_tap")
        env_config.occupancy.dynamic_context_len = load_decoder_context_len_from_checkpoint(args.vae_checkpoint)
        env_config.vae_tap_checkpoint = Path(args.vae_checkpoint)
        env_config.vae_tap_layer = None if args.vae_tap_layer is None else int(args.vae_tap_layer)

    return env_config


def _build_wrapped_env_for_scene(
    *,
    scene,
    env_config: TorchORCAEnvConfig,
    observation_mode: str,
):
    torch_env = TorchORCAEnv(
        scene_factory=lambda template_scene=scene: copy.deepcopy(template_scene),
        config=copy.deepcopy(env_config),
    )

    mode = str(observation_mode).strip().lower()
    if mode == "occupancy":
        gym_env = torch_env
    elif mode == "minimal":
        gym_env = MinimalKinematicsObservationWrapper(torch_env)
    else:
        raise ValueError(f"Unknown --observation-mode: {observation_mode}")

    wrapped_env = wrap_env(gym_env, wrapper="gymnasium", verbose=False)
    return torch_env, wrapped_env


def _load_skrl_models(
    *,
    checkpoint_path: Path,
    observation_space,
    action_space,
    actor_hidden_dims: tuple[int, ...],
    critic_hidden_dims: tuple[int, ...],
    device: torch.device,
    map_extractor_type: str,
    vae_checkpoint: Path | None,
    vae_tap_layer: int | None,
) -> tuple[OccupancyPolicyModel, OccupancyValueModel | OccupancyQValueModel]:
    checkpoint = _load_checkpoint_dict(checkpoint_path=checkpoint_path, device=device)
    policy_state_dict, aux_state_dict, algorithm = _select_policy_and_aux_state_dicts(checkpoint)

    checkpoint_map_extractor_type = _infer_checkpoint_map_extractor_type(checkpoint)
    requested_map_extractor_type = str(map_extractor_type).strip().lower()
    if requested_map_extractor_type != checkpoint_map_extractor_type:
        raise ValueError(
            "Requested map extractor type does not match checkpoint: "
            f"requested='{requested_map_extractor_type}', checkpoint='{checkpoint_map_extractor_type}'."
        )

    tap_bottleneck_hidden_dims: tuple[int, ...] = (128,)
    tap_bottleneck_output_dim = 32
    if checkpoint_map_extractor_type == "vae_tap":
        if vae_checkpoint is None:
            raise ValueError("--vae-checkpoint is required when --map-extractor-type=vae_tap")
        tap_bottleneck_hidden_dims, tap_bottleneck_output_dim = _infer_tap_bottleneck_arch(policy_state_dict)

    policy = OccupancyPolicyModel(
        observation_space,
        action_space,
        device=device,
        hidden_dims=actor_hidden_dims,
        tap_bottleneck_hidden_dims=tap_bottleneck_hidden_dims,
        tap_bottleneck_output_dim=tap_bottleneck_output_dim,
    ).to(device)
    if algorithm == "ppo":
        value_or_critic_model: OccupancyValueModel | OccupancyQValueModel = OccupancyValueModel(
            observation_space,
            action_space,
            device=device,
            hidden_dims=critic_hidden_dims,
            tap_bottleneck_hidden_dims=tap_bottleneck_hidden_dims,
            tap_bottleneck_output_dim=tap_bottleneck_output_dim,
        ).to(device)
    else:
        value_or_critic_model = OccupancyQValueModel(
            observation_space,
            action_space,
            device=device,
            hidden_dims=critic_hidden_dims,
            tap_bottleneck_hidden_dims=tap_bottleneck_hidden_dims,
            tap_bottleneck_output_dim=tap_bottleneck_output_dim,
        ).to(device)

    checkpoint_input_dim = None
    checkpoint_net0 = policy_state_dict.get("net.0.weight")
    if isinstance(checkpoint_net0, torch.Tensor) and checkpoint_net0.ndim == 2:
        checkpoint_input_dim = int(checkpoint_net0.shape[1])
    model_input_dim = None
    if len(policy.net) > 0 and hasattr(policy.net[0], "in_features"):
        model_input_dim = int(policy.net[0].in_features)
    if checkpoint_input_dim is not None and model_input_dim is not None and checkpoint_input_dim != model_input_dim:
        raise ValueError(
            "Model input dimension mismatch before loading weights: "
            f"checkpoint net.0.in_features={checkpoint_input_dim}, model net.0.in_features={model_input_dim}. "
            "This usually means checkpoint and observation settings differ."
        )

    policy.load_state_dict(policy_state_dict)
    value_or_critic_model.load_state_dict(aux_state_dict)
    policy.eval()
    value_or_critic_model.eval()
    return policy, value_or_critic_model


def _policy_action_and_stats(
    *,
    policy: OccupancyPolicyModel,
    value_model: OccupancyValueModel | OccupancyQValueModel,
    state_tensor: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, float, float, float]:
    with torch.no_grad():
        if deterministic:
            mean_actions, log_std_parameter, _ = policy.compute({"states": state_tensor}, role="policy")
            log_std = log_std_parameter.view(1, -1).expand_as(mean_actions)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean_actions, std)
            actions = torch.clamp(mean_actions, -1.0, 1.0)
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            actions, log_prob, _ = policy.act({"states": state_tensor}, role="policy")
            actions = torch.clamp(actions, -1.0, 1.0)
            entropy = policy.get_entropy(role="policy").sum(dim=-1, keepdim=True)

        if isinstance(value_model, OccupancyQValueModel):
            values, _ = value_model.compute(
                {
                    "states": state_tensor,
                    "taken_actions": actions,
                },
                role="critic_1",
            )
        else:
            values, _ = value_model.compute({"states": state_tensor}, role="value")

    value_scalar = float(values.reshape(-1)[0].item())
    log_prob_scalar = float(log_prob.reshape(-1)[0].item())
    entropy_scalar = float(entropy.reshape(-1)[0].item())
    return actions, value_scalar, log_prob_scalar, entropy_scalar


def _compute_fixed_plot_limits(record: EpisodeRecord, dt: float) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = _compute_plot_limits(record, dt=dt)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_span = 0.5 * FIXED_GRID_SIZE_METERS
    return (
        center_x - half_span,
        center_x + half_span,
        center_y - half_span,
        center_y + half_span,
    )


def _run_selected_episode(
    *,
    policy: OccupancyPolicyModel,
    value_model: OccupancyValueModel | OccupancyQValueModel,
    scenes: list,
    selection: ScenarioSelection,
    env_config: TorchORCAEnvConfig,
    deterministic: bool,
    observation_mode: str,
    device: torch.device,
) -> EpisodeRecord:
    scene = _apply_selection_to_scene(scenes[int(selection.scene_index)], selection)
    torch_env, wrapped_env = _build_wrapped_env_for_scene(
        scene=scene,
        env_config=env_config,
        observation_mode=observation_mode,
    )

    try:
        state, _ = wrapped_env.reset()
        if torch_env._goals is None or torch_env._last_positions is None or torch_env._last_velocities is None:
            raise RuntimeError("Environment failed to initialize simulator state")

        controlled_idx = int(torch_env.config.controlled_agent_index)
        executed_positions: list[np.ndarray] = [
            torch_env._last_positions[controlled_idx].detach().cpu().numpy().copy()
        ]
        steps: list[StepRecord] = []

        for _ in range(int(torch_env.config.max_steps)):
            if torch_env._last_positions is None or torch_env._goals is None or torch_env._last_velocities is None:
                raise RuntimeError("Environment state unexpectedly missing during rollout")

            all_positions = torch_env._last_positions.detach().cpu().numpy().copy()
            current_pos = all_positions[controlled_idx].copy()
            current_vel = torch_env._last_velocities[controlled_idx].detach().cpu().numpy().copy()
            goal_pos = torch_env._goals[controlled_idx].detach().cpu().numpy().copy()

            state_tensor = state.to(device=device, dtype=torch.float32)
            actions, value, log_prob, entropy = _policy_action_and_stats(
                policy=policy,
                value_model=value_model,
                state_tensor=state_tensor,
                deterministic=bool(deterministic),
            )

            next_state, reward, terminated, truncated, info = wrapped_env.step(actions)

            if torch_env._last_commanded_velocity is None:
                commanded_velocity = np.zeros(2, dtype=np.float32)
            else:
                commanded_velocity = torch_env._last_commanded_velocity.detach().cpu().numpy().copy()

            reward_scalar = float(torch.as_tensor(reward).reshape(-1)[0].item())
            terminated_flag = bool(torch.as_tensor(terminated).reshape(-1)[0].item())
            truncated_flag = bool(torch.as_tensor(truncated).reshape(-1)[0].item())

            steps.append(
                StepRecord(
                    ego_position=current_pos,
                    current_velocity=current_vel,
                    goal_position=goal_pos,
                    all_positions=all_positions,
                    commanded_velocity=commanded_velocity,
                    value=value,
                    log_prob=log_prob,
                    entropy=entropy,
                    reward=reward_scalar,
                    terminated=terminated_flag,
                    truncated=truncated_flag,
                    info=dict(info),
                )
            )

            if torch_env._last_positions is None:
                raise RuntimeError("Environment positions missing after step")
            executed_positions.append(torch_env._last_positions[controlled_idx].detach().cpu().numpy().copy())

            state = next_state
            if terminated_flag or truncated_flag:
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
    finally:
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SKRL policy rollout in ORCA scenes")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/skrl_ppo_orca_minimal.pt"),
        help="Path to SKRL checkpoint (.pt).",
    )
    parser.add_argument(
        "--template-set",
        choices=["default", "test", "cross", "l_shape", "empty_goal"],
        default="empty_goal",
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

    parser.add_argument("--observation-mode", choices=["occupancy", "minimal"], default="minimal")
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[64, 64])

    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--map-extractor-type", choices=["auto", "conv", "vae_tap"], default="auto")
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional VAE checkpoint used to infer dynamic context length when "
            "--map-extractor-type=vae_tap."
        ),
    )
    parser.add_argument(
        "--vae-tap-layer",
        type=int,
        default=None,
        help="Optional decoder tap layer index when --map-extractor-type=vae_tap.",
    )

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

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

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

    checkpoint = _load_checkpoint_dict(checkpoint_path=Path(args.checkpoint), device=device)
    checkpoint_map_extractor_type = _infer_checkpoint_map_extractor_type(checkpoint)
    requested_map_extractor_type = str(args.map_extractor_type).strip().lower()
    if requested_map_extractor_type == "auto":
        effective_map_extractor_type = checkpoint_map_extractor_type
    elif requested_map_extractor_type != checkpoint_map_extractor_type:
        print(
            "[visualize_skrl_rollout] map extractor mismatch: "
            f"requested={requested_map_extractor_type}, checkpoint={checkpoint_map_extractor_type}. "
            f"Using checkpoint setting: {checkpoint_map_extractor_type}"
        )
        effective_map_extractor_type = checkpoint_map_extractor_type
    else:
        effective_map_extractor_type = requested_map_extractor_type

    env_config = _build_env_config(args, effective_map_extractor_type=effective_map_extractor_type)

    variants_per_scene = int(args.scenario_variants_per_scene)
    total_setups = len(scenes) * variants_per_scene

    initial_setup_index = 0
    selection = _build_scenario_selection(
        scenario_index=initial_setup_index,
        scene_count=len(scenes),
        variants_per_scene=variants_per_scene,
        base_seed=int(args.seed),
    )

    # Build one wrapped env to infer spaces for model initialization.
    probe_scene = _apply_selection_to_scene(scenes[int(selection.scene_index)], selection)
    _probe_torch_env, probe_wrapped_env = _build_wrapped_env_for_scene(
        scene=probe_scene,
        env_config=env_config,
        observation_mode=str(args.observation_mode),
    )
    try:
        policy, value_model = _load_skrl_models(
            checkpoint_path=args.checkpoint,
            observation_space=probe_wrapped_env.observation_space,
            action_space=probe_wrapped_env.action_space,
            actor_hidden_dims=tuple(int(v) for v in args.actor_hidden_dims),
            critic_hidden_dims=tuple(int(v) for v in args.critic_hidden_dims),
            device=device,
            map_extractor_type=effective_map_extractor_type,
            vae_checkpoint=None if args.vae_checkpoint is None else Path(args.vae_checkpoint),
            vae_tap_layer=None if args.vae_tap_layer is None else int(args.vae_tap_layer),
        )
    finally:
        probe_wrapped_env.close()

    record = _run_selected_episode(
        policy=policy,
        value_model=value_model,
        scenes=scenes,
        selection=selection,
        env_config=env_config,
        deterministic=bool(args.deterministic),
        observation_mode=str(args.observation_mode),
        device=device,
    )

    if not record.steps:
        raise RuntimeError("No simulation steps were recorded")

    dt = float(env_config.sim.time_step)
    limits = _compute_fixed_plot_limits(record, dt=dt)
    state: dict[str, object] = {
        "selection": selection,
        "record": record,
        "limits": limits,
    }

    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28)

    setup_prev_ax = fig.add_axes([0.08, 0.18, 0.12, 0.04])
    setup_next_ax = fig.add_axes([0.82, 0.18, 0.12, 0.04])
    setup_label = fig.text(0.22, 0.235, "", fontsize=10)

    slider_ax = fig.add_axes([0.20, 0.10, 0.56, 0.04])
    prev_ax = fig.add_axes([0.08, 0.10, 0.08, 0.04])
    next_ax = fig.add_axes([0.82, 0.10, 0.08, 0.04])

    step_slider = Slider(
        slider_ax,
        "Step",
        0,
        len(record.steps) - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
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
        active_record: EpisodeRecord = state["record"]  # type: ignore[assignment]
        start_xy = active_record.executed_positions[0]
        goal_xy = active_record.steps[0].goal_position
        setup_label.set_text(
            f"Setup {active_record.scenario_index + 1}/{total_setups} | "
            f"scene={active_record.scene_index}, variant={active_record.variant_index}, rollout_seed={active_record.rollout_seed} | "
            f"start=({start_xy[0]:.2f}, {start_xy[1]:.2f}) goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )

    def _render_step(step_idx: int) -> None:
        active_record: EpisodeRecord = state["record"]  # type: ignore[assignment]
        active_limits: tuple[float, float, float, float] = state["limits"]  # type: ignore[assignment]
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
            policy=policy,
            value_model=value_model,
            scenes=scenes,
            selection=selection_local,
            env_config=env_config,
            deterministic=bool(args.deterministic),
            observation_mode=str(args.observation_mode),
            device=device,
        )
        if not record_local.steps:
            raise RuntimeError("Selected setup produced an empty rollout")

        state["selection"] = selection_local
        state["record"] = record_local
        state["limits"] = _compute_fixed_plot_limits(record_local, dt=dt)

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
        active_record: EpisodeRecord = state["record"]  # type: ignore[assignment]
        nxt = max(0, min(len(active_record.steps) - 1, current + delta))
        if nxt != current:
            step_slider.set_val(nxt)

    def move_setup(delta: int) -> None:
        active_selection: ScenarioSelection = state["selection"]  # type: ignore[assignment]
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
