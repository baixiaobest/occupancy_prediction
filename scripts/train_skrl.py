from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Allow running this script directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig
from src.skrl.pipeline import dump_effective_configs, run_skrl_ppo_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORCA policy with SKRL PPO (single environment)")

    parser.add_argument("--template-set", choices=["default", "test", "cross", "l_shape", "empty_goal"], default="default")
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

    parser.add_argument("--map-extractor-type", choices=["conv", "vae_tap"], default="conv")
    parser.add_argument("--vae-checkpoint", type=Path, default=None)

    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--rollouts", type=int, default=1024)
    parser.add_argument("--learning-epochs", type=int, default=8)
    parser.add_argument("--mini-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[256, 256])

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/skrl_ppo_orca.pt"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_config = SkrlEnvBuildConfig(
        template_set=str(args.template_set),
        scene_selection=str(args.scene_selection),
        fixed_scene_index=int(args.fixed_scene_index),
        empty_goal_distance_range=(float(args.empty_goal_distance_range[0]), float(args.empty_goal_distance_range[1])),
        empty_goal_other_agents_range=(int(args.empty_goal_other_agents_range[0]), int(args.empty_goal_other_agents_range[1])),
        empty_goal_other_spawn_radius_range=(
            float(args.empty_goal_other_spawn_radius_range[0]),
            float(args.empty_goal_other_spawn_radius_range[1]),
        ),
        empty_goal_other_goal_distance_range=(
            float(args.empty_goal_other_goal_distance_range[0]),
            float(args.empty_goal_other_goal_distance_range[1]),
        ),
        empty_goal_other_min_start_separation=float(args.empty_goal_other_min_start_separation),
        max_steps=int(args.max_steps),
        controlled_agent_index=int(args.controlled_agent_index),
        max_speed=float(args.max_speed),
        goal_tolerance=float(args.goal_tolerance),
        progress_weight=float(args.progress_weight),
        step_penalty=float(args.step_penalty),
        collision_penalty=float(args.collision_penalty),
        success_reward=float(args.success_reward),
        collision_distance=float(args.collision_distance),
        map_extractor_type=str(args.map_extractor_type),
        vae_checkpoint=None if args.vae_checkpoint is None else Path(args.vae_checkpoint),
    )

    train_config = SkrlPPOTrainConfig(
        total_timesteps=int(args.total_timesteps),
        rollouts=int(args.rollouts),
        learning_epochs=int(args.learning_epochs),
        mini_batches=int(args.mini_batches),
        learning_rate=float(args.learning_rate),
        discount_factor=float(args.discount_factor),
        gae_lambda=float(args.gae_lambda),
        seed=int(args.seed),
        device=str(args.device),
        num_envs=int(args.num_envs),
        actor_hidden_dims=tuple(int(v) for v in args.actor_hidden_dims),
        critic_hidden_dims=tuple(int(v) for v in args.critic_hidden_dims),
        output=Path(args.output),
    )

    effective = dump_effective_configs(env_config, train_config)
    print("[train_skrl] effective env config:", effective["env"])
    print("[train_skrl] effective train config:", effective["train"])
    print(f"[train_skrl] torch.cuda.is_available={torch.cuda.is_available()} | requested device={train_config.device}")

    output_path = run_skrl_ppo_training(env_config=env_config, train_config=train_config)
    print(f"[train_skrl] saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
