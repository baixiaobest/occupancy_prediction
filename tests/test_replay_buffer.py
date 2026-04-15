from __future__ import annotations

import os

import pytest
import torch

from src.rl.replay_buffer import ReplayBuffer


def _make_obs(env_size: int, offset: float = 0.0) -> dict[str, torch.Tensor]:
    # positions: (N_env, N_agents, 2), state_id: (N_env,)
    positions = torch.arange(env_size * 3 * 2, dtype=torch.float32).reshape(env_size, 3, 2)
    positions = positions + float(offset)
    state_id = torch.arange(env_size, dtype=torch.float32) + float(offset)
    return {
        "positions": positions,
        "state_id": state_id,
    }


def test_add_and_sample_with_candidates() -> None:
    buf = ReplayBuffer(capacity=16, seed=123)

    obs = _make_obs(env_size=2, offset=0.0)
    next_obs = _make_obs(env_size=2, offset=100.0)
    actions = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    candidate_actions = torch.randn(2, 4, 8, 2)
    candidate_log_probs = torch.randn(2, 4)

    buf.add_batch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        candidate_actions=candidate_actions,
        candidate_log_probs=candidate_log_probs,
    )

    assert len(buf) == 2

    batch = buf.sample(batch_size=2)
    assert batch.actions.shape == (2, 2)
    assert batch.rewards.shape == (2,)
    assert batch.next_obs["positions"].shape == (2, 3, 2)
    assert batch.candidate_actions is not None
    assert batch.candidate_log_probs is not None
    assert batch.candidate_actions.shape == (2, 4, 8, 2)
    assert batch.candidate_log_probs.shape == (2, 4)


def test_add_multi_env_increases_length_by_env_count() -> None:
    buf = ReplayBuffer(capacity=16, seed=0)

    env_size = 3
    buf.add_batch(
        obs=_make_obs(env_size=env_size, offset=0.0),
        actions=torch.zeros(env_size, 2),
        rewards=torch.zeros(env_size),
        next_obs=_make_obs(env_size=env_size, offset=1.0),
        dones=torch.zeros(env_size),
    )

    assert len(buf) == env_size


def test_capacity_overwrite_keeps_latest_transitions() -> None:
    buf = ReplayBuffer(capacity=3, seed=7)

    # Insert 5 single-env transitions with unique state IDs 0..4.
    for i in range(5):
        obs = {
            "positions": torch.full((1, 2, 2), float(i), dtype=torch.float32),
            "state_id": torch.tensor([float(i)], dtype=torch.float32),
        }
        next_obs = {
            "positions": torch.full((1, 2, 2), float(i + 10), dtype=torch.float32),
            "state_id": torch.tensor([float(i + 10)], dtype=torch.float32),
        }
        buf.add_batch(
            obs=obs,
            actions=torch.tensor([[float(i), float(i)]], dtype=torch.float32),
            rewards=torch.tensor([float(i)], dtype=torch.float32),
            next_obs=next_obs,
            dones=torch.tensor([0.0], dtype=torch.float32),
        )

    assert len(buf) == 3

    sampled = buf.sample(batch_size=3)
    state_ids = set(sampled.obs["state_id"].tolist())
    # Latest three IDs should remain after ring overwrite.
    assert state_ids == {2.0, 3.0, 4.0}


def test_add_batch_raises_on_candidate_k_mismatch() -> None:
    buf = ReplayBuffer(capacity=8, seed=0)

    with pytest.raises(ValueError, match="candidate_log_probs K dimension must match"):
        buf.add_batch(
            obs=_make_obs(env_size=2, offset=0.0),
            actions=torch.zeros(2, 2),
            rewards=torch.zeros(2),
            next_obs=_make_obs(env_size=2, offset=1.0),
            dones=torch.zeros(2),
            candidate_actions=torch.zeros(2, 3, 4, 2),
            candidate_log_probs=torch.zeros(2, 4),
        )


def test_add_batch_raises_on_missing_obs() -> None:
    buf = ReplayBuffer(capacity=8, seed=0)

    with pytest.raises(ValueError, match="obs must not be empty"):
        buf.add_batch(
            obs={},
            actions=torch.zeros(1, 2),
            rewards=torch.zeros(1),
            next_obs=_make_obs(env_size=1, offset=1.0),
            dones=torch.zeros(1),
        )


def test_sample_raises_on_empty_or_oversized_batch() -> None:
    buf = ReplayBuffer(capacity=8, seed=0)

    with pytest.raises(ValueError, match="empty replay buffer"):
        buf.sample(batch_size=1)

    buf.add_batch(
        obs=_make_obs(env_size=1, offset=0.0),
        actions=torch.zeros(1, 2),
        rewards=torch.zeros(1),
        next_obs=_make_obs(env_size=1, offset=1.0),
        dones=torch.zeros(1),
    )

    with pytest.raises(ValueError, match="exceeds buffer size"):
        buf.sample(batch_size=2)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
