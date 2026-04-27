from __future__ import annotations

import copy
from typing import Literal

import gymnasium as gym
import numpy as np

from src.scene import Scene
from src.scene_sampling import make_scene_factory

from .env_orca import ORCASB3Env, ORCASB3EnvConfig

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "stable_baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from exc


class MinimalObsProjectionWrapper(gym.ObservationWrapper):
    """Project dict observations into a compact 6D vector for minimal policy training."""

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


def build_orca_vec_env(
    *,
    scenes: list[Scene],
    selection: str,
    fixed_scene_index: int,
    seed: int,
    num_envs: int,
    env_config: ORCASB3EnvConfig | None = None,
    backend: Literal["dummy", "subproc"] = "dummy",
    minimal_observation: bool = False,
    start_method: str | None = None,
) -> VecEnv:
    """Build a vectorized ORCA SB3 environment.

    The returned object is an SB3 VecEnv (DummyVecEnv or SubprocVecEnv).
    """
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")
    if not scenes:
        raise ValueError("scenes must not be empty")
    if selection not in {"random", "cycle", "fixed"}:
        raise ValueError("selection must be one of: random, cycle, fixed")
    if backend not in {"dummy", "subproc"}:
        raise ValueError("backend must be one of: dummy, subproc")

    base_config = ORCASB3EnvConfig() if env_config is None else copy.deepcopy(env_config)

    env_fns = []
    for rank in range(int(num_envs)):
        env_scene_factory = make_scene_factory(
            scenes,
            selection=selection,
            fixed_scene_index=fixed_scene_index,
            seed=int(seed) + 10007 * rank,
        )
        env_seed = int(seed) + rank
        env_cfg = copy.deepcopy(base_config)

        def _init(
            scene_factory=env_scene_factory,
            config=env_cfg,
            rank_seed=env_seed,
            use_minimal_obs=minimal_observation,
        ) -> gym.Env:
            env = ORCASB3Env(scene_factory=scene_factory, config=config)
            if use_minimal_obs:
                env = MinimalObsProjectionWrapper(env)
            env.action_space.seed(rank_seed)
            env.observation_space.seed(rank_seed)
            return env

        env_fns.append(_init)

    if backend == "dummy":
        vec_env: VecEnv = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)

    vec_env.seed(int(seed))
    return vec_env


__all__ = [
    "MinimalObsProjectionWrapper",
    "build_orca_vec_env",
]
