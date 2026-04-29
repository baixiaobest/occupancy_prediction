from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import datetime as dt

import torch

# Allow running this script directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig, SkrlSACTrainConfig
from src.skrl.pipeline import dump_effective_configs, run_skrl_ppo_training, run_skrl_sac_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORCA policy with SKRL")

    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")

    parser.add_argument("--template-set", choices=["default", "test", "cross", "l_shape", "empty_goal"], default="empty_goal")
    parser.add_argument("--scene-selection", choices=["random", "cycle", "fixed"], default="random")
    parser.add_argument("--fixed-scene-index", type=int, default=0)
    parser.add_argument("--observation-mode", choices=["occupancy", "minimal"], default="occupancy")

    parser.add_argument("--map-extractor-type", choices=["conv", "vae_tap"], default="conv")
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--vae-tap-layer", type=int, default=None)

    parser.add_argument("--total-timesteps", type=int, default=300000)
        
    parser.add_argument("--summary-interval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable Weights & Biases logging",
    )
    parser.add_argument("--wandb-project", type=str, default="occupancy-prediction-rl")
    parser.add_argument("--wandb-run-name", type=str, default=dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["torch_dummy"], default="torch_dummy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, defaults to checkpoints/skrl_<algorithm>_orca_<observation_mode>.pt",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algorithm = str(args.algorithm).strip().lower()
    observation_mode = str(args.observation_mode).strip().lower()
    if args.output is None:
        time = dt.datetime.now().strftime("%m-%d_%H-%M")
        output_path = Path(f"checkpoints/skrl_{algorithm}_orca_{observation_mode}_{time}.pt")
    else:
        output_path = Path(args.output)

    env_config = SkrlEnvBuildConfig(
        template_set=str(args.template_set),
        scene_selection=str(args.scene_selection),
        fixed_scene_index=int(args.fixed_scene_index),
        observation_mode=str(args.observation_mode),
        empty_goal_distance_range=(0.5, 6.0),
        empty_goal_other_agents_range=(0, 0),
        empty_goal_other_spawn_radius_range=(6.0, 6.0),
        empty_goal_other_goal_distance_range=(6.0, 6.0),
        empty_goal_other_min_start_separation=0.8,
        max_steps=200,
        controlled_agent_index=0,
        map_extractor_type=str(args.map_extractor_type),
        vae_checkpoint=None if args.vae_checkpoint is None else Path(args.vae_checkpoint),
        vae_tap_layer=None if args.vae_tap_layer is None else int(args.vae_tap_layer),
    )

    common_train_kwargs = dict(
        total_timesteps=int(args.total_timesteps),
        seed=int(args.seed),
        device=str(args.device),
        num_envs=int(args.num_envs),
        vec_env_backend=str(args.vec_env),
        summary_interval_episodes=int(args.summary_interval_episodes),
        checkpoint_interval=int(args.checkpoint_interval),
        wandb=bool(args.wandb),
        wandb_project=str(args.wandb_project),
        wandb_run_name=args.wandb_run_name,
        output=output_path,
    )

    if algorithm == "sac":
        train_config = SkrlSACTrainConfig(**common_train_kwargs)
        train_fn = run_skrl_sac_training
    else:
        train_config = SkrlPPOTrainConfig(**common_train_kwargs)
        train_fn = run_skrl_ppo_training

    effective = dump_effective_configs(env_config, train_config)
    print("[train_skrl] effective env config:", effective["env"])
    print("[train_skrl] effective train config:", effective["train"])
    print(f"[train_skrl] torch.cuda.is_available={torch.cuda.is_available()} | requested device={train_config.device}")

    output_path = train_fn(env_config=env_config, train_config=train_config)
    print(f"[train_skrl] saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
