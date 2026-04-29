from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class SkrlEnvBuildConfig:
    template_set: str = "default"
    scene_selection: str = "random"
    fixed_scene_index: int = 0
    observation_mode: str = "occupancy"

    empty_goal_distance_range: tuple[float, float] = (2.0, 6.0)
    empty_goal_other_agents_range: tuple[int, int] = (0, 0)
    empty_goal_other_spawn_radius_range: tuple[float, float] = (1.5, 6.0)
    empty_goal_other_goal_distance_range: tuple[float, float] = (2.0, 6.0)
    empty_goal_other_min_start_separation: float = 0.8

    max_steps: int = 200
    controlled_agent_index: int = 0

    map_extractor_type: str = "conv"
    vae_checkpoint: Path | None = None
    vae_tap_layer: int | None = None


@dataclass
class SkrlTrainConfigBase:
    total_timesteps: int = 300000
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_envs: int = 1
    vec_env_backend: str = "torch_dummy"

    initial_policy_std: float = 0.3
    max_policy_std: float = 0.5
    actor_hidden_dims: tuple[int, ...] = (64, 64)
    critic_hidden_dims: tuple[int, ...] = (64, 64)
    tap_bottleneck_hidden_dims: tuple[int, ...] = (128,)
    tap_bottleneck_output_dim: int = 32

    summary_interval_episodes: int = 10

    # Periodic checkpoint saving via SKRL experiment hooks.
    checkpoint_interval: int = 50000

    # Optional Weights & Biases integration through SKRL experiment hooks.
    wandb: bool = False
    wandb_project: str = "occupancy-prediction-rl"
    wandb_run_name: str | None = None

    output: Path = Path("checkpoints/skrl_agent.pt")


@dataclass
class SkrlPPOTrainConfig(SkrlTrainConfigBase):
    rollouts: int = 16000
    learning_epochs: int = 10
    mini_batches: int = 32
    gae_lambda: float = 0.95

    # Policy initialization / PPO control terms
    ratio_clip: float = 0.2
    kl_threshold: float = 0.01
    entropy_loss_scale: float = 0.001

    output: Path = Path("checkpoints/skrl_ppo_orca.pt")


@dataclass
class SkrlSACTrainConfig(SkrlTrainConfigBase):
    memory_size: int = 3.5e4
    gradient_steps: int = 1
    batch_size: int = 512
    polyak: float = 0.005

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4

    random_timesteps: int = 10000
    learning_starts: int = 30000

    learn_entropy: bool = True
    entropy_learning_rate: float = 1e-4
    initial_entropy_value: float = 0.2
    target_entropy: float | None = None

    output: Path = Path("checkpoints/skrl_sac_orca.pt")
