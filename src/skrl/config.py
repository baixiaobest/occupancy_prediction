from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class SkrlEnvBuildConfig:
    template_set: str = "default"
    scene_selection: str = "random"
    fixed_scene_index: int = 0

    empty_goal_distance_range: tuple[float, float] = (2.0, 6.0)
    empty_goal_other_agents_range: tuple[int, int] = (0, 0)
    empty_goal_other_spawn_radius_range: tuple[float, float] = (1.5, 6.0)
    empty_goal_other_goal_distance_range: tuple[float, float] = (2.0, 6.0)
    empty_goal_other_min_start_separation: float = 0.8

    max_steps: int = 200
    controlled_agent_index: int = 0
    max_speed: float = 3.0
    goal_tolerance: float = 0.2

    progress_weight: float = 1.0
    step_penalty: float = 0.0
    collision_penalty: float = -1.0
    success_reward: float = 5.0
    collision_distance: float = 0.4

    map_extractor_type: str = "conv"
    vae_checkpoint: Path | None = None


@dataclass
class SkrlPPOTrainConfig:
    total_timesteps: int = 300000
    rollouts: int = 1024
    learning_epochs: int = 8
    mini_batches: int = 8
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_envs: int = 1

    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)

    output: Path = Path("checkpoints/skrl_ppo_orca.pt")
