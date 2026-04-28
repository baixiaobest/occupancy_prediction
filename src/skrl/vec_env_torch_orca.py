from __future__ import annotations

import copy
from typing import Any, Callable, Literal

import gymnasium as gym
import torch

from src.scene import Scene
from src.scene_sampling import make_scene_factory

from .env_torch_orca import TorchORCAEnv, TorchORCAEnvConfig
from .observation_wrappers import MinimalKinematicsObservationWrapper
from .training_summary import PeriodicEpisodeSummaryWrapper


class TorchDummyVecEnv:
    """Torch-native vector env over a list of single-env instances.

    Keeps observations/rewards/done flags as torch tensors end-to-end.
    """

    def __init__(self, env_fns: list) -> None:
        self.envs = tuple(fn() for fn in env_fns)
        if not self.envs:
            raise ValueError("env_fns must not be empty")

        self.num_envs = int(len(self.envs))
        self.num_agents = 1
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.device = getattr(self.envs[0], "device", torch.device("cpu"))

    def _alloc_obs_batch(self, sample: Any) -> Any:
        if isinstance(sample, dict):
            return {key: self._alloc_obs_batch(value) for key, value in sample.items()}
        tensor = sample if torch.is_tensor(sample) else torch.as_tensor(sample)
        return torch.empty((self.num_envs,) + tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)

    @staticmethod
    def _write_obs(batch: Any, index: int, value: Any) -> None:
        if isinstance(batch, dict):
            for key in batch.keys():
                TorchDummyVecEnv._write_obs(batch[key], index, value[key])
            return

        if torch.is_tensor(value):
            casted = value.to(device=batch.device, dtype=batch.dtype)
        else:
            casted = torch.as_tensor(value, device=batch.device, dtype=batch.dtype)
        batch[index].copy_(casted)

    def _accumulate_info(self, merged: dict[str, Any], info: dict[str, Any], index: int) -> None:
        for key, value in info.items():
            if isinstance(value, bool):
                if key not in merged:
                    merged[key] = torch.zeros(self.num_envs, dtype=torch.bool)
                merged[key][index] = bool(value)
                continue

            if isinstance(value, (int, float)):
                if key not in merged:
                    merged[key] = torch.full((self.num_envs,), float("nan"), dtype=torch.float32)
                merged[key][index] = float(value)
                continue

            if key not in merged:
                merged[key] = [None] * self.num_envs
            merged[key][index] = value

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs_batch: Any | None = None
        merged_info: dict[str, Any] = {}
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else int(seed) + i
            obs, info = env.reset(seed=env_seed, options=options)
            if obs_batch is None:
                obs_batch = self._alloc_obs_batch(obs)
            self._write_obs(obs_batch, i, obs)
            self._accumulate_info(merged_info, info if isinstance(info, dict) else {}, i)

        if obs_batch is None:
            raise RuntimeError("reset produced no observations")
        return obs_batch, merged_info

    def step(self, actions):
        if torch.is_tensor(actions):
            action_batch = actions.detach()
        else:
            action_batch = torch.as_tensor(actions, dtype=torch.float32)

        if action_batch.ndim == 1:
            action_batch = action_batch.unsqueeze(0)

        next_obs_batch: Any | None = None
        rewards = torch.empty(self.num_envs, dtype=torch.float32)
        terminated = torch.empty(self.num_envs, dtype=torch.bool)
        truncated = torch.empty(self.num_envs, dtype=torch.bool)
        merged_info: dict[str, Any] = {}

        for i, env in enumerate(self.envs):
            action_i = action_batch[i]
            obs, reward, term, trunc, info = env.step(action_i)

            done = bool(term or trunc)
            if done:
                final_obs = obs
                reset_obs, reset_info = env.reset()
                info_dict = dict(info) if isinstance(info, dict) else {}
                info_dict["final_observation"] = final_obs
                info_dict["reset_info"] = reset_info
                obs = reset_obs
                info = info_dict

            if next_obs_batch is None:
                next_obs_batch = self._alloc_obs_batch(obs)
            self._write_obs(next_obs_batch, i, obs)
            rewards[i] = float(reward)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            self._accumulate_info(merged_info, info if isinstance(info, dict) else {}, i)

        if next_obs_batch is None:
            raise RuntimeError("step produced no observations")

        return (
            next_obs_batch,
            rewards,
            terminated,
            truncated,
            merged_info,
        )

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def build_torch_orca_vec_env(
    *,
    scenes: list[Scene],
    selection: str,
    fixed_scene_index: int,
    seed: int,
    num_envs: int,
    env_config: TorchORCAEnvConfig,
    observation_mode: str,
    interval_episodes: int,
    backend: Literal["torch_dummy"] = "torch_dummy",
    summary_callback: Callable[[dict[str, object]], None] | None = None,
) -> gym.Env:
    """Build a torch-native vectorized Torch ORCA env for SKRL training."""
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")
    if not scenes:
        raise ValueError("scenes must not be empty")
    if selection not in {"random", "cycle", "fixed"}:
        raise ValueError("selection must be one of: random, cycle, fixed")
    if backend != "torch_dummy":
        raise ValueError("backend must be 'torch_dummy'")

    mode = str(observation_mode).strip().lower()
    if mode not in {"occupancy", "minimal"}:
        raise ValueError(
            f"Unknown observation_mode '{observation_mode}'. Expected one of: occupancy, minimal"
        )

    base_config = copy.deepcopy(env_config)
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
            rank_idx=rank,
        ) -> gym.Env:
            env: gym.Env = TorchORCAEnv(scene_factory=scene_factory, config=config)
            if mode == "minimal":
                env = MinimalKinematicsObservationWrapper(env)
            env = PeriodicEpisodeSummaryWrapper(
                env,
                interval_episodes=int(interval_episodes),
                prefix=f"[train_skrl][env={int(rank_idx)}]",
                summary_key=f"env_{int(rank_idx)}",
                summary_callback=summary_callback,
            )
            env.action_space.seed(rank_seed)
            env.observation_space.seed(rank_seed)
            return env

        env_fns.append(_init)

    vec_env = TorchDummyVecEnv(env_fns)

    return vec_env


__all__ = [
    "TorchDummyVecEnv",
    "build_torch_orca_vec_env",
]
