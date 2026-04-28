from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

from src.experiment_utils import EmptyGoalTemplateConfig, build_scene_pool, seed_everything
from src.scene import Scene
from src.scene_sampling import make_scene_factory
from src.sb3.utils import load_decoder_context_len_from_checkpoint
from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig
from src.skrl.env_torch_orca import (
    TorchORCAEnv,
    TorchORCAEnvConfig,
    TorchORCARewardConfig,
    TorchORCASimConfig,
)
from src.skrl.models import (
    OccupancyPolicyModel,
    OccupancyValueModel,
    VAEDecoderTapFeatureExtractor,
)
from src.skrl.observation_wrappers import MinimalKinematicsObservationWrapper
from src.skrl.training_summary import PeriodicEpisodeSummaryWrapper, install_agent_tracking_summary
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
    )

    if str(env_cfg.map_extractor_type) == "vae_tap":
        if env_cfg.vae_checkpoint is None:
            raise ValueError("vae_checkpoint is required when map_extractor_type='vae_tap'")
        config.occupancy.dynamic_context_len = load_decoder_context_len_from_checkpoint(env_cfg.vae_checkpoint)

    return config


def run_skrl_ppo_training(
    env_config: SkrlEnvBuildConfig,
    train_config: SkrlPPOTrainConfig,
) -> Path:
    """Train SKRL PPO agent on one or many ORCA environments."""
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

    if num_envs == 1:
        env = _make_single_env(env_config, seed=int(train_config.seed), device=device)
        env = PeriodicEpisodeSummaryWrapper(
            env,
            interval_episodes=summary_interval_episodes,
            prefix="[train_skrl]",
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
        )

    wrapped_env = wrap_env(env, wrapper="gymnasium", verbose=False)

    map_extractor_type = str(env_config.map_extractor_type).strip().lower()
    shared_feature_extractor = None
    if map_extractor_type == "vae_tap":
        if str(env_config.observation_mode).strip().lower() != "occupancy":
            raise ValueError("map_extractor_type='vae_tap' requires observation_mode='occupancy'")
        if env_config.vae_checkpoint is None:
            raise ValueError("vae_checkpoint is required when map_extractor_type='vae_tap'")
        shared_feature_extractor = VAEDecoderTapFeatureExtractor(
            observation_space=wrapped_env.observation_space,
            vae_checkpoint=str(env_config.vae_checkpoint),
            tap_layer=(None if env_config.vae_tap_layer is None else int(env_config.vae_tap_layer)),
            tap_bottleneck_hidden_dim=tuple(int(v) for v in train_config.tap_bottleneck_hidden_dims),
            tap_bottleneck_output_dim=int(train_config.tap_bottleneck_output_dim),
        ).to(device)
        shared_feature_extractor.eval()

    models = {
        "policy": OccupancyPolicyModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.actor_hidden_dims),
            initial_std=float(train_config.initial_policy_std),
            max_std=float(train_config.max_policy_std),
            feature_extractor=shared_feature_extractor,
        ),
        "value": OccupancyValueModel(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            device=device,
            hidden_dims=tuple(int(v) for v in train_config.critic_hidden_dims),
            feature_extractor=shared_feature_extractor,
        ),
    }

    memory = RandomMemory(
        memory_size=int(train_config.rollouts),
        num_envs=int(wrapped_env.num_envs),
        device=device,
    )

    output_path = Path(train_config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    agent_cfg["experiment"]["wandb"] = False
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
    finally:
        wrapped_env.close()

    return output_path


def dump_effective_configs(env_config: SkrlEnvBuildConfig, train_config: SkrlPPOTrainConfig) -> dict[str, dict]:
    return {
        "env": asdict(env_config),
        "train": {
            **asdict(train_config),
            "output": str(train_config.output),
        },
    }
