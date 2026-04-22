from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualize_model import build_models
from src.rl.training.training_app import RLTrainingApp
from src.rl.training.training_profiler import RunProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL selection/Q network with ORCA scenes")
    parser.add_argument("--mode", choices=["simple_state_q", "counterfactual_q"], default="simple_state_q")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None, help="Path to trained VAE checkpoint")
    parser.add_argument("--empty-goal-distance-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--template-set", choices=["default", "test", "cross", "l_shape"], default="default")
    parser.add_argument("--scene-selection", choices=["random", "cycle", "fixed"], default="random")
    parser.add_argument("--fixed-scene-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--collect-steps-per-iter", type=int, default=8)
    parser.add_argument("--updates-per-iter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--num-bootstrap-candidates", type=int, default=8)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-max-speed", type=float, default=2.0)
    parser.add_argument("--candidate-delta-std", type=float, default=0.25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--target-update-interval", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--simple-q-hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--simple-proposal-hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--simple-proposal-min-variance", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--loss-type", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--tap-layer", type=int, default=1)
    parser.add_argument("--occupancy-resolution", type=float, default=0.1)
    parser.add_argument("--env-max-steps", type=int, default=200)
    parser.add_argument("--controlled-agent-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/rl_q_selection.pt"))
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-scene-selection", choices=["random", "cycle", "fixed"], default="cycle")
    parser.add_argument("--eval-seed-offset", type=int, default=100000)
    parser.add_argument("--profile", action="store_true", help="Enable built-in phase timing and cProfile output")
    parser.add_argument("--profile-top-n", type=int, default=30, help="Number of cProfile rows to print")
    parser.add_argument("--profile-output", type=Path, default=None, help="Optional path to save raw cProfile stats")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable Weights & Biases logging",
    )
    parser.add_argument("--wandb-project", type=str, default="occupancy-prediction-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiler = RunProfiler(
        enabled=bool(args.profile),
        top_n=int(args.profile_top_n),
        output_path=args.profile_output,
    )
    app = RLTrainingApp(args, profiler=profiler, decoder_builder=build_models)
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
