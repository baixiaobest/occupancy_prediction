from __future__ import annotations

import copy
import importlib
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

from src.experiment_utils import EmptyGoalTemplateConfig, build_scene_pool, seed_everything
from src.scene import Scene
from src.scene_sampling import make_scene_factory
from src.sb3.utils import load_decoder_context_len_from_checkpoint
from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig, SkrlSACTrainConfig, SkrlTrainConfigBase
from src.skrl.env_torch_orca import (
    TorchORCAEnv,
    TorchORCAEnvConfig,
    TorchORCARewardConfig,
    TorchORCASimConfig,
)
from src.skrl.models import (
    OccupancyPolicyModel,
    OccupancyQValueModel,
    OccupancyValueModel,
    build_tap_bottleneck_feature_projector,
)
from src.skrl.observation_wrappers import MinimalKinematicsObservationWrapper
from src.skrl.custom_sac import CustomSAC
from src.skrl.training_summary import (
    PeriodicEpisodeSummaryWrapper,
    _build_wandb_summary_callback,
    install_agent_tracking_summary,
)
from src.skrl.vec_env_torch_orca import build_torch_orca_vec_env


def _build_scene_pool(config: SkrlEnvBuildConfig, *, seed: int) -> list[Scene]:
    empty_goal_cfg: EmptyGoalTemplateConfig | None = None
    if config.template_set == "empty_goal":
        empty_goal_cfg = EmptyGoalTemplateConfig(
            goal_distance_range=tuple(float(v) for v in config.empty_goal_distance_range),
            goal_seed=int(seed),
            num_other_agents_range=tuple(int(v) for v in config.empty_goal_other_agents_range),
            other_agent_spawn_radius_range=tuple(float(v) for v in config.empty_goal_other_spawn_radius_range),
            other_agent_goal_distance_range=tuple(float(v) for v in config.empty_goal_other_goal_distance_range),
            other_agent_min_start_separation=float(config.empty_goal_other_min_start_separation),
        )
    return build_scene_pool(str(config.template_set), empty_goal=empty_goal_cfg)


def _make_single_env(env_cfg: SkrlEnvBuildConfig, seed: int, device: torch.device) -> gym.Env:
    config = _build_torch_orca_env_config(env_cfg, device=device)

    scenes = _build_scene_pool(env_cfg, seed=seed)
    scene_factory = make_scene_factory(
        scenes,
        selection=str(env_cfg.scene_selection),
        fixed_scene_index=int(env_cfg.fixed_scene_index),
        seed=int(seed),
    )

    base_env = TorchORCAEnv(scene_factory=scene_factory, config=config)

    observation_mode = str(env_cfg.observation_mode).strip().lower()
    if observation_mode == "occupancy":
        return base_env
    if observation_mode == "minimal":
        return MinimalKinematicsObservationWrapper(base_env)
    raise ValueError(
        f"Unknown observation_mode '{env_cfg.observation_mode}'. Expected one of: occupancy, minimal"
    )


def _build_torch_orca_env_config(
    env_cfg: SkrlEnvBuildConfig,
    *,
    device: torch.device,
) -> TorchORCAEnvConfig:
    sim_cfg = TorchORCASimConfig()
    reward_cfg = TorchORCARewardConfig()
    config = TorchORCAEnvConfig(
        max_steps=int(env_cfg.max_steps),
        controlled_agent_index=int(env_cfg.controlled_agent_index),
        device=str(device),
        sim=sim_cfg,
        reward=reward_cfg,
        map_extractor_type=str(env_cfg.map_extractor_type),
        vae_tap_checkpoint=(None if env_cfg.vae_checkpoint is None else Path(env_cfg.vae_checkpoint)),
        vae_tap_layer=(None if env_cfg.vae_tap_layer is None else int(env_cfg.vae_tap_layer)),
    )

    if str(env_cfg.map_extractor_type) == "vae_tap":
        if env_cfg.vae_checkpoint is None:
            raise ValueError("vae_checkpoint is required when map_extractor_type='vae_tap'")
        config.occupancy.dynamic_context_len = load_decoder_context_len_from_checkpoint(env_cfg.vae_checkpoint)

    return config


def _build_wandb_experiment_kwargs(train_config: SkrlTrainConfigBase, *, output_path: Path) -> dict[str, object]:
    run_name = train_config.wandb_run_name
    if run_name is None or not str(run_name).strip():
        run_name = output_path.stem

    return {
        "project": str(train_config.wandb_project),
        "name": str(run_name),
    }


def _save_checkpoint_file_to_wandb(*, checkpoint_path: Path, wandb_module: object | None = None) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found for W&B upload: {checkpoint_path}")

    module = wandb_module
    if module is None:
        try:
            module = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError("wandb is required when wandb logging is enabled. Install with: pip install wandb") from exc

    module.save(str(checkpoint_path), base_path=str(checkpoint_path.parent), policy="now")


def _install_periodic_checkpoint_upload_hook(
    *,
    agent,
    train_config: SkrlTrainConfigBase,
) -> None:
    if not bool(train_config.wandb):
        return

    if not hasattr(agent, "write_tracking_data") or not hasattr(agent, "experiment_dir"):
        return

    original_write_tracking_data = agent.write_tracking_data
    checkpoint_dir = Path(str(agent.experiment_dir)) / "checkpoints"
    wandb_module = importlib.import_module("wandb")

    uploaded_checkpoint_paths: set[str] = set()

    def _patched_write_tracking_data(timestep: int, timesteps: int) -> None:
        original_write_tracking_data(timestep, timesteps)

        for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
            checkpoint_key = str(checkpoint_path.resolve())
            if checkpoint_key in uploaded_checkpoint_paths:
                continue
            _save_checkpoint_file_to_wandb(checkpoint_path=checkpoint_path, wandb_module=wandb_module)
            uploaded_checkpoint_paths.add(checkpoint_key)

    agent.write_tracking_data = _patched_write_tracking_data


def _save_checkpoint_to_wandb_if_enabled(*, train_config: SkrlTrainConfigBase, checkpoint_path: Path) -> None:
    if not bool(train_config.wandb):
        return
    try:
        _save_checkpoint_file_to_wandb(checkpoint_path=checkpoint_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to upload checkpoint to W&B: {checkpoint_path}") from exc


def _prepare_wrapped_env(
    *,
    env_config: SkrlEnvBuildConfig,
    train_config: SkrlTrainConfigBase,
) -> tuple[object, torch.device, bool]:
    num_envs = int(train_config.num_envs)
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")

    seed_everything(int(train_config.seed))

    device = torch.device(str(train_config.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    summary_interval_episodes = int(train_config.summary_interval_episodes)
    if summary_interval_episodes <= 0:
        raise ValueError("summary_interval_episodes must be > 0")

    enable_wandb = bool(train_config.wandb)
    summary_callback = _build_wandb_summary_callback(train_config=train_config)

    if num_envs == 1:
        env = _make_single_env(env_config, seed=int(train_config.seed), device=device)
        env = PeriodicEpisodeSummaryWrapper(
            env,
            interval_episodes=summary_interval_episodes,
            prefix="[train_skrl]",
            summary_key="env_0",
            summary_callback=summary_callback,
        )
    else:
        env = build_torch_orca_vec_env(
            scenes=_build_scene_pool(env_config, seed=int(train_config.seed)),
            selection=str(env_config.scene_selection),
            fixed_scene_index=int(env_config.fixed_scene_index),
            seed=int(train_config.seed),
            num_envs=num_envs,
            env_config=_build_torch_orca_env_config(env_config, device=device),
            observation_mode=str(env_config.observation_mode),
            interval_episodes=summary_interval_episodes,
            backend=str(train_config.vec_env_backend),
            summary_callback=summary_callback,
        )

    wrapped_env = wrap_env(env, wrapper="gymnasium", verbose=False)
    return wrapped_env, device, enable_wandb


def _prepare_output_path(train_config: SkrlTrainConfigBase) -> Path:
    output_path = Path(train_config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _run_training(
    *,
    agent,
    wrapped_env,
    train_config: SkrlTrainConfigBase,
    output_path: Path,
) -> Path:
    trainer_cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
    trainer_cfg["timesteps"] = int(train_config.total_timesteps)
    trainer_cfg["headless"] = True
    trainer_cfg["disable_progressbar"] = False

    trainer = SequentialTrainer(
        env=wrapped_env,
        agents=agent,
        cfg=trainer_cfg,
    )

    try:
        trainer.train()
        agent.save(str(output_path))
        _save_checkpoint_to_wandb_if_enabled(train_config=train_config, checkpoint_path=output_path)
    finally:
        wrapped_env.close()

    return output_path


def run_skrl_ppo_training(
    env_config: SkrlEnvBuildConfig,
    train_config: SkrlPPOTrainConfig,
) -> Path:
    """Train SKRL PPO agent on one or many ORCA environments."""
    wrapped_env, device, enable_wandb = _prepare_wrapped_env(
        env_config=env_config,
        train_config=train_config,
    )

    shared_tap_projector = build_tap_bottleneck_feature_projector(
        wrapped_env.observation_space,
        tap_bottleneck_hidden_dims=tuple(int(v) for v in train_config.tap_bottleneck_hidden_dims),
        tap_bottleneck_output_dim=int(train_config.tap_bottleneck_output_dim),
    )

    models = {
        "policy": OccupancyPolicyModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.actor_hidden_dims),
            initial_std=float(train_config.initial_policy_std),
            max_std=float(train_config.max_policy_std),
            tap_projector=shared_tap_projector,
        ),
        "value": OccupancyValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            tap_projector=shared_tap_projector,
        ),
    }

    memory = RandomMemory(
        memory_size=int(train_config.rollouts),
        num_envs=int(wrapped_env.num_envs),
        device=device,
    )

    output_path = _prepare_output_path(train_config)

    agent_cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    agent_cfg["rollouts"] = int(train_config.rollouts)
    agent_cfg["learning_epochs"] = int(train_config.learning_epochs)
    agent_cfg["mini_batches"] = int(train_config.mini_batches)
    agent_cfg["learning_rate"] = float(train_config.learning_rate)
    agent_cfg["discount_factor"] = float(train_config.discount_factor)
    agent_cfg["lambda"] = float(train_config.gae_lambda)
    agent_cfg["ratio_clip"] = float(train_config.ratio_clip)
    agent_cfg["kl_threshold"] = float(train_config.kl_threshold)
    agent_cfg["entropy_loss_scale"] = float(train_config.entropy_loss_scale)
    agent_cfg["random_timesteps"] = 0
    agent_cfg["learning_starts"] = 0
    agent_cfg["experiment"]["wandb"] = enable_wandb
    if enable_wandb:
        agent_cfg["experiment"]["wandb_kwargs"] = _build_wandb_experiment_kwargs(
            train_config,
            output_path=output_path,
        )
    # Align tracking writes with update cadence to avoid mostly-empty summary lines.
    agent_cfg["experiment"]["write_interval"] = int(train_config.rollouts)
    checkpoint_interval = int(train_config.checkpoint_interval)
    if checkpoint_interval > 0:
        agent_cfg["experiment"]["checkpoint_interval"] = checkpoint_interval
        agent_cfg["experiment"]["directory"] = str(output_path.parent)
        agent_cfg["experiment"]["experiment_name"] = output_path.stem
        agent_cfg["experiment"]["store_separately"] = False

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device,
        cfg=agent_cfg,
    )
    install_agent_tracking_summary(agent, prefix="[train_skrl]")
    _install_periodic_checkpoint_upload_hook(agent=agent, train_config=train_config)

    return _run_training(
        agent=agent,
        wrapped_env=wrapped_env,
        train_config=train_config,
        output_path=output_path,
    )


def run_skrl_sac_training(
    env_config: SkrlEnvBuildConfig,
    train_config: SkrlSACTrainConfig,
) -> Path:
    """Train SKRL SAC agent on one or many ORCA environments."""
    if int(train_config.train_freq) <= 0:
        raise ValueError("train_freq must be > 0")

    wrapped_env, device, enable_wandb = _prepare_wrapped_env(
        env_config=env_config,
        train_config=train_config,
    )

    shared_tap_projector = build_tap_bottleneck_feature_projector(
        wrapped_env.observation_space,
        tap_bottleneck_hidden_dims=tuple(int(v) for v in train_config.tap_bottleneck_hidden_dims),
        tap_bottleneck_output_dim=int(train_config.tap_bottleneck_output_dim),
    )

    models = {
        "policy": OccupancyPolicyModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.actor_hidden_dims),
            initial_std=float(train_config.initial_policy_std),
            max_std=float(train_config.max_policy_std),
            tap_projector=shared_tap_projector,
        ),
        "critic_1": OccupancyQValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            tap_projector=shared_tap_projector,
        ),
        "critic_2": OccupancyQValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            tap_projector=shared_tap_projector,
        ),
        "target_critic_1": OccupancyQValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            tap_projector=shared_tap_projector,
        ),
        "target_critic_2": OccupancyQValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            tap_projector=shared_tap_projector,
        ),
    }

    memory = RandomMemory(
        memory_size=int(train_config.memory_size),
        num_envs=int(wrapped_env.num_envs),
        device=device,
    )

    output_path = _prepare_output_path(train_config)

    agent_cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    agent_cfg["gradient_steps"] = int(train_config.gradient_steps)
    agent_cfg["batch_size"] = int(train_config.batch_size)
    agent_cfg["discount_factor"] = float(train_config.discount_factor)
    agent_cfg["polyak"] = float(train_config.polyak)
    agent_cfg["actor_learning_rate"] = float(train_config.actor_learning_rate)
    agent_cfg["critic_learning_rate"] = float(train_config.critic_learning_rate)
    agent_cfg["random_timesteps"] = int(train_config.random_timesteps)
    agent_cfg["learning_starts"] = int(train_config.learning_starts)
    agent_cfg["train_freq"] = int(train_config.train_freq)
    agent_cfg["learn_entropy"] = bool(train_config.learn_entropy)
    agent_cfg["entropy_learning_rate"] = float(train_config.entropy_learning_rate)
    agent_cfg["initial_entropy_value"] = float(train_config.initial_entropy_value)
    agent_cfg["target_entropy"] = (
        None if train_config.target_entropy is None else float(train_config.target_entropy)
    )
    agent_cfg["experiment"]["wandb"] = enable_wandb
    if enable_wandb:
        agent_cfg["experiment"]["wandb_kwargs"] = _build_wandb_experiment_kwargs(
            train_config,
            output_path=output_path,
        )
    # Keep SAC write cadence reasonable for off-policy updates.
    agent_cfg["experiment"]["write_interval"] = int(max(train_config.batch_size, 1000))
    checkpoint_interval = int(train_config.checkpoint_interval)
    if checkpoint_interval > 0:
        agent_cfg["experiment"]["checkpoint_interval"] = checkpoint_interval
        agent_cfg["experiment"]["directory"] = str(output_path.parent)
        agent_cfg["experiment"]["experiment_name"] = output_path.stem
        agent_cfg["experiment"]["store_separately"] = False

    agent = CustomSAC(
        models=models,
        memory=memory,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device,
        cfg=agent_cfg,
    )
    _install_periodic_checkpoint_upload_hook(agent=agent, train_config=train_config)

    return _run_training(
        agent=agent,
        wrapped_env=wrapped_env,
        train_config=train_config,
        output_path=output_path,
    )


def dump_effective_configs(env_config: SkrlEnvBuildConfig, train_config: SkrlTrainConfigBase) -> dict[str, dict]:
    return {
        "env": asdict(env_config),
        "train": {
            **asdict(train_config),
            "output": str(train_config.output),
        },
    }
