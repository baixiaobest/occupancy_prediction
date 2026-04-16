from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from src.rl.counterfactual import (
    integrate_velocity_plans,
    rollout_counterfactual_futures,
    sample_random_velocity_plans,
)


class DummyDecoder(nn.Module):
    def forward(
        self,
        z: torch.Tensor,
        dynamic_context: torch.Tensor,
        static_x: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
        current_position_offset: torch.Tensor | None = None,
        tap_layer: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del z, static_x
        batch_size = dynamic_context.shape[0]
        height, width = dynamic_context.shape[-2:]

        vel_x = current_velocity[:, 0].view(batch_size, 1, 1, 1, 1)
        logits = vel_x.expand(batch_size, 1, 2, height, width).contiguous()

        if tap_layer is None:
            return logits

        pos_x = current_position_offset[:, 0].view(batch_size, 1, 1, 1)
        tap = pos_x.expand(batch_size, 3, height // 2, width // 2).contiguous()
        return logits, tap


def test_sample_random_velocity_plans_shape_and_nominal_candidate() -> None:
    current_velocity = torch.tensor([[0.5, -0.2], [0.1, 0.3]], dtype=torch.float32)
    plans = sample_random_velocity_plans(
        current_velocity,
        num_candidates=4,
        horizon=5,
        max_speed=1.0,
        delta_std=0.3,
        dt=0.1,
    )

    assert plans.shape == (2, 4, 5, 2)
    assert torch.allclose(plans[:, 0], current_velocity[:, None, :])
    assert torch.all(torch.linalg.vector_norm(plans, dim=-1) <= 1.0 + 1e-5)


def test_integrate_velocity_plans_cumulative_offsets() -> None:
    plans = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]]]],
        dtype=torch.float32,
    )
    offsets = integrate_velocity_plans(plans, dt=0.2)

    expected = torch.tensor(
        [[[[0.2, 0.0], [0.4, 0.0], [0.5, 0.0]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(offsets, expected)


def test_rollout_counterfactual_futures_shapes_and_offsets() -> None:
    decoder = DummyDecoder()
    dynamic_context = torch.zeros((2, 1, 4, 8, 8), dtype=torch.float32)
    static_map = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    candidate_plans = torch.tensor(
        [
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]],
            ],
            [
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]],
            ],
        ],
        dtype=torch.float32,
    )

    rollout = rollout_counterfactual_futures(
        decoder=decoder,
        dynamic_context=dynamic_context,
        static_map=static_map,
        candidate_velocity_plans=candidate_plans,
        latent_channels=4,
        latent_shape=(1, 2, 2),
        dt=0.1,
        tap_layer=1,
    )

    assert rollout.candidate_velocity_plans.shape == (2, 2, 3, 2)
    assert rollout.candidate_position_offsets.shape == (2, 2, 3, 2)
    assert rollout.predicted_logits.shape == (2, 2, 3, 1, 8, 8)
    assert rollout.tapped_features is not None
    assert rollout.tapped_features.shape == (2, 2, 3, 3, 4, 4)

    expected_offsets = torch.tensor(
        [
            [
                [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
                [[0.05, 0.0], [0.1, 0.0], [0.15, 0.0]],
            ],
            [
                [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
                [[0.0, 0.05], [0.0, 0.1], [0.0, 0.15]],
            ],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rollout.candidate_position_offsets, expected_offsets)

    flattened_plans, flattened_taps = rollout.flatten_candidates_for_q()
    assert flattened_plans.shape == (4, 3, 2)
    assert flattened_taps is not None
    assert flattened_taps.shape == (4, 3, 3, 4, 4)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
