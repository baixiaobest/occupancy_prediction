from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib
import os
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

# Allow running this script directly from the repository root.
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
from sb3.minimal_policy import MinimalActorCriticPolicy
from sb3.policy import OccupancyActorCriticPolicy
from src.training_profiler import RunProfiler

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an occupancy-aware ORCA SB3 PPO policy")

    default_wandb_name = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser.add_argument(
        "--template-set",
        choices=["default", "test", "cross", "l_shape", "empty_goal"],
        default="default",
    )
    parser.add_argument("--scene-selection", choices=["random", "cycle", "fixed"], default="random")
    parser.add_argument("--fixed-scene-index", type=int, default=0)
    parser.add_argument("--empty-goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--empty-goal-other-agents-range", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--empty-goal-other-spawn-radius-range", type=float, nargs=2, default=[1.5, 6.0])
    parser.add_argument("--empty-goal-other-goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--empty-goal-other-min-start-separation", type=float, default=0.8)

    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--controlled-agent-index", type=int, default=0)
    parser.add_argument("--max-speed", type=float, default=3.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.2)

    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--collision-penalty", type=float, default=-1.0)
    parser.add_argument("--success-reward", type=float, default=5.0)
    parser.add_argument("--collision-distance", type=float, default=0.4)

    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--policy", choices=["occupancy", "minimal"], default="occupancy")
    parser.add_argument("--map-extractor-type", choices=["conv", "vae_tap"], default="conv")
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--vae-tap-layer", type=int, default=1)

    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sb3_ppo_orca"))

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="occupancy-prediction-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=default_wandb_name)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="train_ppo")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-upload-model-interval", type=int, default=10000)

    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-top-n", type=int, default=40)
    parser.add_argument("--profile-output", type=Path, default=Path("profiles/train_ppo.prof"))

    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    seed: int,
    empty_goal_other_agents_range: tuple[int, int],
    empty_goal_other_spawn_radius_range: tuple[float, float],
    empty_goal_other_goal_distance_range: tuple[float, float],
    empty_goal_other_min_start_separation: float,
) -> list[Scene]:
    if template_set == "empty_goal":
        templates = empty_goal_templates(
            goal_distance_range=goal_distance_range,
            goal_seed=seed,
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


def _wandb_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, tuple):
            config[key] = list(value)
        else:
            config[key] = value
    return config


def _init_wandb(args: argparse.Namespace):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    original_sys_path = list(sys.path)
    sys.path = [path for path in sys.path if os.path.abspath(path) != repo_root]

    sys.modules.pop("wandb", None)
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        sys.path = original_sys_path
        raise ImportError("wandb is required for --wandb. Install with: pip install wandb") from exc
    except Exception:
        sys.path = original_sys_path
        raise

    sys.path = original_sys_path

    if not hasattr(wandb, "init"):
        raise ImportError(
            "Imported 'wandb' does not expose init(). "
            "A local workspace module may be shadowing the pip package."
        )

    run = wandb.init(
        project=str(args.wandb_project),
        entity=None if args.wandb_entity is None else str(args.wandb_entity),
        name=str(args.wandb_name),
        group=None if args.wandb_group is None else str(args.wandb_group),
        job_type=str(args.wandb_job_type),
        tags=None if args.wandb_tags is None else [str(tag) for tag in args.wandb_tags],
        config=_wandb_config_from_args(args),
        sync_tensorboard=True,
    )
    return wandb, run


def _resolved_model_file_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else Path(f"{path}.zip")


def _load_decoder_context_len_from_checkpoint(checkpoint_path: Path) -> int:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("VAE checkpoint must be a dict")
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError("VAE checkpoint must contain dict key 'model_config'")

    if "decoder_context_len" not in model_cfg:
        raise ValueError(
            "VAE checkpoint model_config must contain key 'decoder_context_len'"
        )

    context_len = int(model_cfg["decoder_context_len"])

    return context_len


class _MinimalObsWrapper(gym.ObservationWrapper):
    """Project dict observation to a compact 6D vector for minimal policy training."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        goal = np.asarray(observation["goal_position"], dtype=np.float32).reshape(2)
        current_velocity = np.asarray(observation["current_velocity"], dtype=np.float32).reshape(2)
        last_commanded_velocity = np.asarray(observation["last_commanded_velocity"], dtype=np.float32).reshape(2)
        return np.concatenate([goal, current_velocity, last_commanded_velocity], axis=0).astype(np.float32, copy=False)


class _WandbModelUploadCallback(BaseCallback):
    def __init__(self, *, wandb_module: Any, output_path: Path, interval_steps: int) -> None:
        super().__init__(verbose=0)
        interval = int(interval_steps)
        if interval <= 0:
            raise ValueError("wandb model upload interval must be > 0")
        self.wandb_module = wandb_module
        self.output_path = Path(output_path)
        self.interval_steps = interval
        self.next_upload_step = interval

    def _save_and_upload(self) -> None:
        checkpoint_base = self.output_path.with_name(f"{self.output_path.name}_step_{self.num_timesteps}")
        self.model.save(str(checkpoint_base))
        checkpoint_file = _resolved_model_file_path(checkpoint_base)
        self.wandb_module.save(str(checkpoint_file))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_upload_step:
            self._save_and_upload()
            while self.num_timesteps >= self.next_upload_step:
                self.next_upload_step += self.interval_steps
        return True


class _RewardBreakdownLoggingCallback(BaseCallback):
    """Aggregate reward term breakdown over rollout and log once per rollout."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._term_sums: dict[str, float] = {}
        self._num_samples: int = 0

    def _on_rollout_start(self) -> None:
        self._term_sums.clear()
        self._num_samples = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, (list, tuple)):
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            reward_terms = info.get("reward_terms")
            if not isinstance(reward_terms, dict):
                continue

            self._num_samples += 1
            for key, value in reward_terms.items():
                try:
                    term_value = float(value)
                except (TypeError, ValueError):
                    continue
                self._term_sums[key] = self._term_sums.get(key, 0.0) + term_value
        return True

    def _on_rollout_end(self) -> None:
        if self._num_samples <= 0 or not self._term_sums:
            return

        inv_n = 1.0 / float(self._num_samples)
        for key, summed in self._term_sums.items():
            metric_key = f"reward_terms/{key}"
            mean_value = float(summed) * inv_n
            self.logger.record(metric_key, mean_value)


def main() -> None:
    args = parse_args()

    profiler = RunProfiler(
        enabled=bool(args.profile),
        top_n=int(args.profile_top_n),
        output_path=args.profile_output if bool(args.profile) else None,
        log_fn=lambda message: print(message, flush=True),
    )
    profiler.start()
    wandb_module = None
    wandb_run = None

    _seed_everything(int(args.seed))

    tensorboard_log_path = args.tensorboard_log
    if bool(args.wandb) and tensorboard_log_path is None:
        tensorboard_log_path = Path("runs/sb3_ppo")

    other_agents_range = (
        int(args.empty_goal_other_agents_range[0]),
        int(args.empty_goal_other_agents_range[1]),
    )
    if other_agents_range[0] < 0 or other_agents_range[1] < 0:
        raise ValueError("--empty-goal-other-agents-range values must be >= 0")
    if float(args.empty_goal_other_min_start_separation) < 0.0:
        raise ValueError("--empty-goal-other-min-start-separation must be >= 0")

    goal_distance_range = (float(args.empty_goal_distance_range[0]), float(args.empty_goal_distance_range[1]))
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
        goal_distance_range=goal_distance_range,
        seed=int(args.seed),
        empty_goal_other_agents_range=other_agents_range,
        empty_goal_other_spawn_radius_range=other_spawn_radius_range,
        empty_goal_other_goal_distance_range=other_goal_distance_range,
        empty_goal_other_min_start_separation=float(args.empty_goal_other_min_start_separation),
    )

    scene_factory = _make_scene_factory(
        scenes,
        selection=str(args.scene_selection),
        fixed_scene_index=int(args.fixed_scene_index),
        seed=int(args.seed),
    )

    sim_cfg = ORCASB3SimConfig(
        max_speed=float(args.max_speed),
        goal_tolerance=float(args.goal_tolerance),
    )
    reward_cfg = ORCASB3RewardConfig(
        progress_weight=float(args.progress_weight),
        step_penalty=float(args.step_penalty),
        collision_penalty=float(args.collision_penalty),
        success_reward=float(args.success_reward),
        collision_distance=float(args.collision_distance),
    )
    env_cfg = ORCASB3EnvConfig(
        max_steps=int(args.max_steps),
        controlled_agent_index=int(args.controlled_agent_index),
        sim=sim_cfg,
        reward=reward_cfg,
    )

    policy_name = str(args.policy)
    if policy_name == "occupancy" and str(args.map_extractor_type) == "vae_tap":
        if args.vae_checkpoint is None:
            raise ValueError("--vae-checkpoint is required when --map-extractor-type vae_tap")
        env_cfg.occupancy.dynamic_context_len = _load_decoder_context_len_from_checkpoint(args.vae_checkpoint)

    env = ORCASB3Env(scene_factory=scene_factory, config=env_cfg)

    policy_kwargs: dict[str, Any] = {
        "actor_hidden_dims": [int(v) for v in args.actor_hidden_dims],
        "critic_hidden_dims": [int(v) for v in args.critic_hidden_dims],
        "actor_activation_fn": torch.nn.Tanh,
        "critic_activation_fn": torch.nn.Tanh,
    }
    if policy_name == "minimal":
        env = _MinimalObsWrapper(env)
        policy_cls: type = MinimalActorCriticPolicy
    else:
        policy_cls = OccupancyActorCriticPolicy
        map_extractor_type = str(args.map_extractor_type)
        policy_kwargs["map_extractor_type"] = map_extractor_type
        if map_extractor_type == "vae_tap":
            if args.vae_checkpoint is None:
                raise ValueError("--vae-checkpoint is required when --map-extractor-type vae_tap")
            policy_kwargs["vae_checkpoint"] = str(args.vae_checkpoint)
            policy_kwargs["vae_tap_layer"] = (
                None if args.vae_tap_layer is None else int(args.vae_tap_layer)
            )
        else:
            policy_kwargs["map_conv_channels"] = [8, 8, 16, 16, 32, 32]

    model = PPO(
        policy=policy_cls,
        env=env,
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        verbose=1,
        seed=int(args.seed),
        device=str(args.device),
        tensorboard_log=None if tensorboard_log_path is None else str(tensorboard_log_path),
        policy_kwargs=policy_kwargs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        if bool(args.wandb):
            wandb_module, wandb_run = _init_wandb(args)

        callback_list: list[BaseCallback] = [
            _RewardBreakdownLoggingCallback()
        ]

        upload_interval = int(args.wandb_upload_model_interval)
        if wandb_module is not None and upload_interval > 0:
            callback_list.append(
                _WandbModelUploadCallback(
                    wandb_module=wandb_module,
                    output_path=args.output,
                    interval_steps=upload_interval,
                )
            )

        learn_callback: BaseCallback | None
        if not callback_list:
            learn_callback = None
        elif len(callback_list) == 1:
            learn_callback = callback_list[0]
        else:
            learn_callback = CallbackList(callback_list)

        with profiler.section("learn"):
            model.learn(
                total_timesteps=int(args.total_timesteps),
                progress_bar=bool(args.progress_bar),
                callback=learn_callback,
            )

        model.save(str(args.output))
        print(f"Saved PPO model to {args.output}")

        if wandb_module is not None:
            wandb_module.save(str(_resolved_model_file_path(args.output)))
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        profiler.stop()
        profiler.report()


if __name__ == "__main__":
    main()
