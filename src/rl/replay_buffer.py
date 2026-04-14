from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch


TensorDict = dict[str, torch.Tensor]


@dataclass
class ReplaySampleBatch:
    """A sampled minibatch from replay memory.

    All tensor fields are batched along dimension 0.
    """

    obs: TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: TensorDict
    dones: torch.Tensor
    candidate_actions: torch.Tensor | None = None
    candidate_log_probs: torch.Tensor | None = None


class ReplayBuffer:
    """Replay buffer that accepts environment batches and stores per-transition items.

    The insertion API always expects an environment axis, so single-environment
    usage should pass tensors with leading dimension 1.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self._rng = random.Random(int(seed))
        self._items: list[dict[str, Any]] = []
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._next_idx = 0

    def add_batch(
        self,
        *,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: TensorDict,
        dones: torch.Tensor,
        candidate_actions: torch.Tensor | None = None,
        candidate_log_probs: torch.Tensor | None = None,
    ) -> None:
        """Insert one environment batch of transitions.

        Args:
            obs: Dictionary of observation tensors, each shaped (N_env, ...).
            actions: Selected actions, shaped (N_env, ...).
            rewards: Step rewards, shaped (N_env,).
            next_obs: Next-step observation dictionary, each shaped (N_env, ...).
            dones: Done flags, shaped (N_env,).
            candidate_actions: Optional candidate action sets, shaped (N_env, K, ...).
            candidate_log_probs: Optional candidate log-probs, shaped (N_env, K).
        """
        if len(obs) == 0:
            raise ValueError("obs must not be empty")
        if len(next_obs) == 0:
            raise ValueError("next_obs must not be empty")

        env_size = self._infer_env_size(obs=obs, next_obs=next_obs)
        actions_t = self._to_tensor(actions, name="actions")
        rewards_t = self._to_tensor(rewards, name="rewards")
        dones_t = self._to_tensor(dones, name="dones")

        if actions_t.ndim < 1 or actions_t.shape[0] != env_size:
            raise ValueError("actions must have shape (N_env, ...)")
        if rewards_t.ndim != 1 or rewards_t.shape[0] != env_size:
            raise ValueError("rewards must have shape (N_env,)")
        if dones_t.ndim != 1 or dones_t.shape[0] != env_size:
            raise ValueError("dones must have shape (N_env,)")

        cand_actions_t: torch.Tensor | None = None
        if candidate_actions is not None:
            cand_actions_t = self._to_tensor(candidate_actions, name="candidate_actions")
            if cand_actions_t.ndim < 2 or cand_actions_t.shape[0] != env_size:
                raise ValueError("candidate_actions must have shape (N_env, K, ...)")

        cand_log_probs_t: torch.Tensor | None = None
        if candidate_log_probs is not None:
            cand_log_probs_t = self._to_tensor(candidate_log_probs, name="candidate_log_probs")
            if cand_log_probs_t.ndim != 2 or cand_log_probs_t.shape[0] != env_size:
                raise ValueError("candidate_log_probs must have shape (N_env, K)")
            if cand_actions_t is not None and cand_log_probs_t.shape[1] != cand_actions_t.shape[1]:
                raise ValueError("candidate_log_probs K dimension must match candidate_actions")

        for env_idx in range(env_size):
            item = {
                "obs": self._slice_obs(obs, env_idx),
                "actions": actions_t[env_idx].detach().cpu(),
                "rewards": rewards_t[env_idx].detach().cpu(),
                "next_obs": self._slice_obs(next_obs, env_idx),
                "dones": dones_t[env_idx].detach().cpu(),
                "candidate_actions": None if cand_actions_t is None else cand_actions_t[env_idx].detach().cpu(),
                "candidate_log_probs": None
                if cand_log_probs_t is None
                else cand_log_probs_t[env_idx].detach().cpu(),
            }
            self._append(item)

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> ReplaySampleBatch:
        """Uniformly sample a minibatch of transitions."""
        if len(self._items) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        bsz = int(batch_size)
        if bsz <= 0:
            raise ValueError("batch_size must be > 0")
        if bsz > len(self._items):
            raise ValueError(
                f"Requested batch_size={bsz} exceeds buffer size {len(self._items)}"
            )

        indices = self._rng.sample(range(len(self._items)), k=bsz)
        rows = [self._items[i] for i in indices]

        obs = self._stack_obs([row["obs"] for row in rows], device=device)
        actions = self._stack_tensor_rows([row["actions"] for row in rows], device=device)
        rewards = self._stack_tensor_rows([row["rewards"] for row in rows], device=device).to(dtype=torch.float32)
        next_obs = self._stack_obs([row["next_obs"] for row in rows], device=device)
        dones = self._stack_tensor_rows([row["dones"] for row in rows], device=device).to(dtype=torch.float32)

        has_candidate_actions = all(row["candidate_actions"] is not None for row in rows)
        has_candidate_log_probs = all(row["candidate_log_probs"] is not None for row in rows)

        candidate_actions = None
        if has_candidate_actions:
            candidate_actions = self._stack_tensor_rows(
                [row["candidate_actions"] for row in rows],
                device=device,
            )

        candidate_log_probs = None
        if has_candidate_log_probs:
            candidate_log_probs = self._stack_tensor_rows(
                [row["candidate_log_probs"] for row in rows],
                device=device,
            )

        return ReplaySampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            candidate_actions=candidate_actions,
            candidate_log_probs=candidate_log_probs,
        )

    @staticmethod
    def _to_tensor(value: torch.Tensor | Any, *, name: str) -> torch.Tensor:
        t = torch.as_tensor(value)
        if t.ndim == 0:
            raise ValueError(f"{name} must include an environment axis")
        return t

    @staticmethod
    def _infer_env_size(*, obs: TensorDict, next_obs: TensorDict) -> int:
        env_size: int | None = None

        def _check_dict(d: TensorDict, dict_name: str) -> None:
            nonlocal env_size
            for key, value in d.items():
                t = torch.as_tensor(value)
                if t.ndim < 1:
                    raise ValueError(f"{dict_name}[{key}] must have shape (N_env, ...)")
                if env_size is None:
                    env_size = int(t.shape[0])
                elif int(t.shape[0]) != env_size:
                    raise ValueError(f"Inconsistent N_env in {dict_name}[{key}]")

        _check_dict(obs, "obs")
        _check_dict(next_obs, "next_obs")

        if env_size is None:
            raise ValueError("Could not infer environment size")
        return env_size

    @staticmethod
    def _slice_obs(obs: TensorDict, env_idx: int) -> TensorDict:
        out: TensorDict = {}
        for key, value in obs.items():
            t = torch.as_tensor(value)
            out[key] = t[env_idx].detach().cpu()
        return out

    @staticmethod
    def _stack_obs(obs_rows: list[TensorDict], device: torch.device | str | None) -> TensorDict:
        keys = list(obs_rows[0].keys())
        for row in obs_rows[1:]:
            if list(row.keys()) != keys:
                raise ValueError("Observation keys are inconsistent across sampled transitions")

        out: TensorDict = {}
        for key in keys:
            stacked = torch.stack([row[key] for row in obs_rows], dim=0)
            if device is not None:
                stacked = stacked.to(device=device)
            out[key] = stacked
        return out

    @staticmethod
    def _stack_tensor_rows(
        rows: list[torch.Tensor],
        device: torch.device | str | None,
    ) -> torch.Tensor:
        stacked = torch.stack(rows, dim=0)
        if device is not None:
            stacked = stacked.to(device=device)
        return stacked

    def _append(self, item: dict[str, Any]) -> None:
        if len(self._items) < self.capacity:
            self._items.append(item)
            return

        self._items[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self.capacity


__all__ = ["ReplayBuffer", "ReplaySampleBatch", "TensorDict"]
