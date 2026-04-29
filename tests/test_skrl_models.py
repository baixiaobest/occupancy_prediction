from __future__ import annotations

import importlib

import numpy as np
import torch

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")
pytest.importorskip("skrl")

import gymnasium as gym

from src.skrl.models import OccupancyPolicyModel, OccupancyValueModel, _build_mlp, _flatten_states


def test_build_mlp_output_shape() -> None:
    mlp = _build_mlp(input_dim=4, output_dim=2, hidden_dims=(8, 8))
    x = torch.randn(5, 4)
    y = mlp(x)
    assert tuple(y.shape) == (5, 2)


def test_policy_model_computes_mean_and_log_std() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    model = OccupancyPolicyModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        hidden_dims=(16, 16),
        initial_std=0.5,
    )
    mean, log_std, extra = model.compute({"states": torch.randn(3, 4)}, role="policy")

    assert tuple(mean.shape) == (3, 2)
    assert tuple(log_std.shape) == (2,)
    assert isinstance(extra, dict)
    assert torch.allclose(torch.exp(log_std), torch.full((2,), 0.5), atol=1e-4)


def test_policy_model_rejects_nonpositive_initial_std() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    with pytest.raises(ValueError, match="initial_std must be > 0"):
        OccupancyPolicyModel(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            initial_std=0.0,
        )


def test_policy_model_rejects_invalid_max_std() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    with pytest.raises(ValueError, match="max_std must be > 1e-6"):
        OccupancyPolicyModel(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            max_std=1e-6,
        )


def test_policy_model_squashes_std_to_fixed_and_configured_bounds() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    model = OccupancyPolicyModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        hidden_dims=(16, 16),
        max_std=0.3,
    )

    with torch.no_grad():
        model.std_parameter.fill_(100.0)
    _, log_std_hi, _ = model.compute({"states": torch.randn(3, 4)}, role="policy")
    std_hi = torch.exp(log_std_hi)
    assert torch.all(std_hi <= torch.full_like(std_hi, 0.3 + 1e-4))

    with torch.no_grad():
        model.std_parameter.fill_(-100.0)
    _, log_std_lo, _ = model.compute({"states": torch.randn(3, 4)}, role="policy")
    std_lo = torch.exp(log_std_lo)
    assert torch.all(std_lo >= torch.full_like(std_lo, 1e-6 - 1e-8))


def test_value_model_output_shape() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    model = OccupancyValueModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        hidden_dims=(16, 16),
    )
    values, extra = model.compute({"states": torch.randn(7, 4)}, role="value")

    assert tuple(values.shape) == (7, 1)
    assert isinstance(extra, dict)


def test_flatten_states_dict_uses_sorted_key_order() -> None:
    states = {
        "z": torch.tensor([[9.0, 8.0]], dtype=torch.float32),
        "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    flat = _flatten_states(states)
    assert tuple(flat.shape) == (1, 4)
    assert torch.allclose(flat, torch.tensor([[1.0, 2.0, 9.0, 8.0]], dtype=torch.float32))


def test_policy_and_value_apply_tap_bottleneck_projector() -> None:
    observation_space = gym.spaces.Dict(
        {
            "decoder_tap": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
            "goal_position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "current_velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "last_commanded_velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    policy = OccupancyPolicyModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        hidden_dims=(8,),
        tap_bottleneck_hidden_dims=(16,),
        tap_bottleneck_output_dim=6,
    )
    value = OccupancyValueModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        hidden_dims=(8,),
        tap_bottleneck_hidden_dims=(16,),
        tap_bottleneck_output_dim=6,
    )

    x = {
        "decoder_tap": torch.randn(3, 10),
        "goal_position": torch.randn(3, 2),
        "current_velocity": torch.randn(3, 2),
        "last_commanded_velocity": torch.randn(3, 2),
    }
    mean, log_std, _ = policy.compute({"states": x}, role="policy")
    values, _ = value.compute({"states": x}, role="value")

    assert tuple(mean.shape) == (3, 2)
    assert tuple(log_std.shape) == (2,)
    assert tuple(values.shape) == (3, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
