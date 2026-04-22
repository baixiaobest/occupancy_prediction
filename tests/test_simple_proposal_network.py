from __future__ import annotations

import os

import pytest
import torch

from src.rl.networks.simple_proposal_network import SimpleVelocityProposalNetwork


def test_simple_proposal_network_forward_shapes() -> None:
    model = SimpleVelocityProposalNetwork(horizon=4, hidden_dims=(32, 32), min_variance=1e-5)
    current_velocity = torch.tensor([[0.0, 0.1], [0.2, -0.3]], dtype=torch.float32)
    goal_relative_position = torch.tensor([[1.0, 0.0], [-0.5, 0.8]], dtype=torch.float32)

    delta_mean, delta_var = model(
        current_velocity=current_velocity,
        goal_relative_position=goal_relative_position,
    )

    assert tuple(delta_mean.shape) == (2, 4, 2)
    assert tuple(delta_var.shape) == (2, 4, 2)
    assert torch.all(delta_var > 0.0)


def test_simple_proposal_network_sample_actions_with_current_velocity_candidate() -> None:
    model = SimpleVelocityProposalNetwork(horizon=3, hidden_dims=(), min_variance=1e-8)
    current_velocity = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    goal_relative_position = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    # Make output deterministic enough for test expectations.
    with torch.no_grad():
        model.delta_velocity_mean_head.weight.zero_()
        model.delta_velocity_mean_head.bias.zero_()
        model.delta_velocity_var_head.weight.zero_()
        model.delta_velocity_var_head.bias.fill_(-50.0)

    actions = model.sample_actions(
        current_velocity=current_velocity,
        goal_relative_position=goal_relative_position,
        num_candidates=4,
        include_current_velocity_candidate=True,
        generator=torch.Generator().manual_seed(0),
        max_speed=0.5,
    )

    assert tuple(actions.shape) == (1, 4, 2)
    speeds = torch.linalg.vector_norm(actions, dim=-1)
    assert torch.all(speeds <= 0.50001)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
