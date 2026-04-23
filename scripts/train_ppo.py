from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

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
from sb3.policy import OccupancyActorCriticPolicy

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an occupancy-aware ORCA SB3 PPO policy")

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


def main() -> None:
    args = parse_args()
    _seed_everything(int(args.seed))

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

    env = ORCASB3Env(scene_factory=scene_factory, config=env_cfg)

    policy_kwargs = {
        "actor_hidden_dims": [int(v) for v in args.actor_hidden_dims],
        "critic_hidden_dims": [int(v) for v in args.critic_hidden_dims],
        "actor_activation_fn": torch.nn.Tanh,
        "critic_activation_fn": torch.nn.Tanh,
        "map_conv_channels": [16, 32, 64, 64],
    }

    model = PPO(
        policy=OccupancyActorCriticPolicy,
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
        tensorboard_log=None if args.tensorboard_log is None else str(args.tensorboard_log),
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=int(args.total_timesteps),
        progress_bar=bool(args.progress_bar),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved PPO model to {args.output}")


if __name__ == "__main__":
    main()
