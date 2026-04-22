from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from src.rl.counterfactual import CounterfactualRolloutBatch
from src.rl.q_trainers.q_trainer import (
    QTrainerConfig,
    RandomCandidateQTrainer,
    q_scores_to_probabilities,
    sample_action_indices_from_q_scores,
    soft_update_module,
)
from src.rl.replay_buffer import ReplaySampleBatch


class DummyDecoder(nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("DummyDecoder should not be called directly in this test")


class DummyQNetwork(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(
        self,
        goal_position: torch.Tensor,
        current_velocity: torch.Tensor,
        planned_velocities: torch.Tensor,
        tapped_future_features: torch.Tensor,
    ) -> torch.Tensor:
        taps = torch.as_tensor(tapped_future_features, dtype=torch.float32).reshape(planned_velocities.shape[0], -1)
        tap_mean = taps.mean(dim=1)
        plan_x = torch.as_tensor(planned_velocities, dtype=torch.float32)[:, :, 0].sum(dim=1)
        goal_x = torch.as_tensor(goal_position, dtype=torch.float32)[:, 0]
        velocity_x = torch.as_tensor(current_velocity, dtype=torch.float32)[:, 0]
        return self.scale * (plan_x + tap_mean + goal_x + velocity_x)


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


def _dummy_rollout_fn(
    *,
    decoder: nn.Module,
    dynamic_context: torch.Tensor,
    static_map: torch.Tensor,
    candidate_velocity_plans: torch.Tensor,
    latent_channels: int,
    latent_shape: tuple[int, int, int],
    dt: float,
    current_position_offset: torch.Tensor | None = None,
    tap_layer: int | None = None,
    binary_feedback: bool = False,
    threshold: float = 0.5,
    latent_samples: torch.Tensor | None = None,
) -> CounterfactualRolloutBatch:
    del decoder, dynamic_context, static_map, latent_channels, latent_shape, dt
    del current_position_offset, tap_layer, binary_feedback, threshold, latent_samples
    plans = torch.as_tensor(candidate_velocity_plans, dtype=torch.float32)
    taps = plans[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    logits = torch.zeros((*plans.shape[:3], 1, 1, 1), dtype=torch.float32, device=plans.device)
    offsets = torch.cumsum(plans * 0.1, dim=2)
    return CounterfactualRolloutBatch(
        candidate_velocity_plans=plans,
        candidate_position_offsets=offsets,
        predicted_logits=logits,
        tapped_features=taps,
    )


def _make_batch() -> ReplaySampleBatch:
    obs = {
        "dynamic_context": torch.zeros((2, 1, 2, 4, 4), dtype=torch.float32),
        "static_map": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "current_velocity": torch.tensor([[0.5, 0.0], [0.2, 0.0]], dtype=torch.float32),
        "goal_position": torch.tensor([[1.0, 0.0], [0.3, 0.0]], dtype=torch.float32),
    }
    next_obs = {
        "dynamic_context": torch.zeros((2, 1, 2, 4, 4), dtype=torch.float32),
        "static_map": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "current_velocity": torch.tensor([[0.4, 0.0], [0.1, 0.0]], dtype=torch.float32),
        "goal_position": torch.tensor([[0.6, 0.0], [0.2, 0.0]], dtype=torch.float32),
    }
    actions = torch.tensor(
        [
            [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]],
            [[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]],
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


def test_random_candidate_q_trainer_updates_q_and_target_networks() -> None:
    q_network = DummyQNetwork(scale=0.0)
    target_q_network = DummyQNetwork(scale=1.0)
    optimizer = torch.optim.SGD(q_network.parameters(), lr=0.1)
    trainer = RandomCandidateQTrainer(
        q_network=q_network,
        target_q_network=target_q_network,
        decoder=DummyDecoder(),
        optimizer=optimizer,
        config=QTrainerConfig(
            discount=0.5,
            target_tau=0.25,
            selection_temperature=1.0,
            selection_seed=0,
            num_bootstrap_candidates=2,
            max_speed=2.0,
            delta_std=0.0,
            dt=0.1,
            tap_layer=1,
            latent_channels=1,
            latent_shape=(1, 1, 1),
            loss_type="mse",
        ),
        candidate_sampler=_dummy_candidate_sampler,
        counterfactual_rollout_fn=_dummy_rollout_fn,
    )

    stats = trainer.train_step(_make_batch())

    assert stats.loss > 0.0
    assert stats.done_fraction == pytest.approx(0.5)
    assert stats.reward_mean == pytest.approx(0.25)
    assert stats.target_mean == pytest.approx(1.9)
    assert stats.next_q_mean == pytest.approx(3.65)
    assert stats.selection_entropy_mean > 0.0
    assert q_network.scale.item() != pytest.approx(0.0)
    assert target_q_network.scale.item() == pytest.approx(1.0)
    trainer.update_target_network()
    assert target_q_network.scale.item() == pytest.approx(0.75 + 0.25 * q_network.scale.item(), rel=1e-5)


def test_q_score_softmax_selection_helpers() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

    probs = q_scores_to_probabilities(scores, temperature=0.5)
    expected_probs = torch.softmax(scores / 0.5, dim=1)
    assert torch.allclose(probs, expected_probs)

    generator = torch.Generator(device=scores.device)
    generator.manual_seed(0)
    sampled_indices, sampled_probs = sample_action_indices_from_q_scores(
        scores,
        temperature=0.5,
        generator=generator,
    )
    assert torch.equal(sampled_indices, torch.tensor([2], dtype=torch.int64))
    assert torch.allclose(sampled_probs, expected_probs)


def test_soft_update_module_rejects_invalid_tau() -> None:
    source = DummyQNetwork(scale=1.0)
    target = DummyQNetwork(scale=0.0)

    with pytest.raises(ValueError, match=r"tau must be in \[0, 1\]"):
        soft_update_module(source, target, tau=1.5)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))