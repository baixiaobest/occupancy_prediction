from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

import src.rl.q_trainers.simple_q_trainer as simple_q_trainer_module
from src.rl.replay_buffer import ReplaySampleBatch
from src.rl.q_trainers.simple_q_trainer import (
    SimpleActionCandidateQTrainer,
    SimpleQTrainerConfig,
    SimpleRandomCandidateQTrainer,
)


class _DummyQNetwork(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_position: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action_x = torch.as_tensor(action, dtype=torch.float32)[:, 0]
        goal_x = torch.as_tensor(goal_position, dtype=torch.float32)[:, 0]
        velocity_x = torch.as_tensor(current_velocity, dtype=torch.float32)[:, 0]
        return self.scale * (action_x + goal_x + velocity_x)


class _DeterministicProposalNetwork(nn.Module):
    def sample_actions(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_relative_position: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
        num_candidates: int,
        include_current_velocity_candidate: bool = True,
        generator: torch.Generator | None = None,
        max_speed: float | None = None,
    ) -> torch.Tensor:
        del goal_relative_position, goal_position, generator, max_speed
        velocity = torch.as_tensor(current_velocity, dtype=torch.float32)
        batch_size = int(velocity.shape[0])
        candidates = int(num_candidates)
        actions = velocity[:, None, :].expand(-1, candidates, -1).clone()
        for candidate_idx in range(candidates):
            actions[:, candidate_idx, 0] += float(candidate_idx)
        if include_current_velocity_candidate:
            actions[:, 0, :] = velocity
        return actions


def _dummy_candidate_sampler(
    current_velocity: torch.Tensor,
    *,
    num_candidates: int,
    horizon: int,
    max_speed: float,
    delta_std: float = 0.25,
    dt: float = 0.1,
    include_current_velocity_candidate: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    del max_speed, delta_std, dt, include_current_velocity_candidate, generator
    velocity = torch.as_tensor(current_velocity, dtype=torch.float32)
    plans = velocity[:, None, None, :].expand(-1, num_candidates, horizon, -1).clone()
    for candidate_idx in range(num_candidates):
        plans[:, candidate_idx, :, 0] += float(candidate_idx)
    return plans


def _make_batch() -> ReplaySampleBatch:
    obs = {
        "current_velocity": torch.tensor([[0.5, 0.0], [0.2, 0.0]], dtype=torch.float32),
        "goal_position": torch.tensor([[1.0, 0.0], [0.3, 0.0]], dtype=torch.float32),
    }
    next_obs = {
        "current_velocity": torch.tensor([[0.4, 0.0], [0.1, 0.0]], dtype=torch.float32),
        "goal_position": torch.tensor([[0.6, 0.0], [0.2, 0.0]], dtype=torch.float32),
    }
    actions = torch.tensor(
        [
            [0.5, 0.0],
            [0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([1.0, -0.5], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    return ReplaySampleBatch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
    )


def test_simple_random_candidate_q_trainer_updates_q_and_target_networks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simple_q_trainer_module, "sample_random_velocity_plans", _dummy_candidate_sampler)

    q_network = _DummyQNetwork(scale=0.0)
    target_q_network = _DummyQNetwork(scale=1.0)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.1)
    trainer = SimpleRandomCandidateQTrainer(
        q_network=q_network,
        target_q_network=target_q_network,
        optimizer=optimizer,
        config=SimpleQTrainerConfig(
            discount=0.5,
            target_tau=0.25,
            selection_temperature=0.01,
            selection_seed=0,
            num_bootstrap_candidates=2,
            max_speed=2.0,
            delta_std=0.0,
            dt=0.1,
            loss_type="mse",
        ),
    )

    stats = trainer.train_step(_make_batch())

    assert stats.loss > 0.0
    assert stats.q_pred_mean == pytest.approx(0.0)
    assert stats.done_fraction == pytest.approx(0.5)
    assert stats.reward_mean == pytest.approx(0.25)
    assert stats.target_mean == pytest.approx(0.85)
    assert stats.next_q_mean == pytest.approx(1.9)
    assert stats.selection_entropy_mean < 1e-3
    assert q_network.scale.item() != pytest.approx(0.0)
    assert target_q_network.scale.item() == pytest.approx(0.75 + 0.25 * q_network.scale.item(), rel=1e-5)


def test_simple_random_candidate_q_trainer_rejects_missing_goal_position() -> None:
    q_network = _DummyQNetwork(scale=0.0)
    target_q_network = _DummyQNetwork(scale=1.0)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.1)
    trainer = SimpleRandomCandidateQTrainer(
        q_network=q_network,
        target_q_network=target_q_network,
        optimizer=optimizer,
        config=SimpleQTrainerConfig(),
    )

    batch = _make_batch()
    invalid_batch = ReplaySampleBatch(
        obs={"current_velocity": batch.obs["current_velocity"]},
        actions=batch.actions,
        rewards=batch.rewards,
        next_obs=batch.next_obs,
        dones=batch.dones,
    )

    with pytest.raises(ValueError, match=r"missing required keys"):
        trainer.train_step(invalid_batch)


def test_simple_action_candidate_q_trainer_uses_proposal_network_samples() -> None:
    q_network = _DummyQNetwork(scale=0.0)
    target_q_network = _DummyQNetwork(scale=1.0)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.1)
    proposal_network = _DeterministicProposalNetwork()
    trainer = SimpleActionCandidateQTrainer(
        q_network=q_network,
        target_q_network=target_q_network,
        proposal_network=proposal_network,
        optimizer=optimizer,
        config=SimpleQTrainerConfig(
            discount=0.5,
            target_tau=0.25,
            selection_temperature=0.01,
            selection_seed=0,
            num_bootstrap_candidates=2,
            max_speed=2.0,
            delta_std=0.0,
            dt=0.1,
            loss_type="mse",
        ),
    )

    stats = trainer.train_step(_make_batch())

    assert stats.loss > 0.0
    assert stats.target_mean == pytest.approx(0.85)
    assert stats.next_q_mean == pytest.approx(1.9)
    assert q_network.scale.item() != pytest.approx(0.0)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
