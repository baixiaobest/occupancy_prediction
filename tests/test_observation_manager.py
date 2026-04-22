from __future__ import annotations

import os

import pytest
import torch

from src.rl import build_simple_state_observation_config
from src.rl.managers.observation_manager import (
    ObservationBatchContext,
    ObservationManager,
    ObservationTermCfg,
    OnlineOccupancyObservationConfig,
    build_online_occupancy_observation_manager,
)
from src.scene import ObstacleSpec, Scene


def _make_scene() -> Scene:
    return Scene(
        agents=[],
        obstacles=[
            ObstacleSpec(vertices=[(0.1, -0.2), (0.5, -0.2), (0.5, 0.2), (0.1, 0.2)]),
        ],
    )


def _make_observation_context(
    *,
    other_agent_x: float,
    controlled_velocity: tuple[float, float] = (0.2, 0.0),
    goal_xy: tuple[float, float] = (1.0, 0.0),
) -> ObservationBatchContext:
    raw_obs = {
        "positions": torch.tensor([[[0.0, 0.0], [other_agent_x, 0.0]]], dtype=torch.float32),
        "velocities": torch.tensor([[[controlled_velocity[0], controlled_velocity[1]], [0.0, 0.0]]], dtype=torch.float32),
        "goals": torch.tensor([[[goal_xy[0], goal_xy[1]], [0.0, 0.0]]], dtype=torch.float32),
        "controlled_agent_index": torch.tensor([0], dtype=torch.int64),
    }
    return ObservationBatchContext(raw_obs=raw_obs, scene=_make_scene())


def test_online_occupancy_observation_manager_outputs_expected_terms() -> None:
    manager = build_online_occupancy_observation_manager(
        OnlineOccupancyObservationConfig(
            decoder_context_len=2,
            local_map_shape=(12, 10),
            occupancy_resolution=(0.2, 0.2),
            agent_radius=0.2,
        )
    )

    obs = manager.compute(_make_observation_context(other_agent_x=0.8))

    assert tuple(obs["dynamic_context"].shape) == (1, 1, 2, 12, 10)
    assert tuple(obs["static_map"].shape) == (1, 1, 12, 10)
    assert tuple(obs["current_velocity"].shape) == (1, 2)
    assert tuple(obs["goal_position"].shape) == (1, 2)

    assert torch.allclose(obs["current_velocity"], torch.tensor([[0.2, 0.0]], dtype=torch.float32))
    assert torch.allclose(obs["goal_position"], torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    assert obs["static_map"].sum().item() > 0.0
    assert torch.allclose(obs["dynamic_context"][0, 0, 0], obs["dynamic_context"][0, 0, 1])


def test_observation_manager_reset_clears_dynamic_history() -> None:
    manager = build_online_occupancy_observation_manager(
        OnlineOccupancyObservationConfig(
            decoder_context_len=2,
            local_map_shape=(12, 10),
            occupancy_resolution=(0.2, 0.2),
            agent_radius=0.2,
        )
    )

    manager.compute(_make_observation_context(other_agent_x=0.8))
    manager.compute(_make_observation_context(other_agent_x=0.2))
    obs_without_reset = manager.compute(_make_observation_context(other_agent_x=-0.4))

    assert not torch.allclose(
        obs_without_reset["dynamic_context"][0, 0, 0],
        obs_without_reset["dynamic_context"][0, 0, 1],
    )

    manager.reset()
    obs_after_reset = manager.compute(_make_observation_context(other_agent_x=-0.4))

    assert torch.allclose(
        obs_after_reset["dynamic_context"][0, 0, 0],
        obs_after_reset["dynamic_context"][0, 0, 1],
    )


def test_observation_manager_raises_on_invalid_term_shape() -> None:
    manager = ObservationManager(
        terms=[
            ObservationTermCfg(
                name="bad",
                fn=lambda context, params: torch.zeros((context.num_envs + 1, 2), dtype=torch.float32),
            )
        ]
    )

    with pytest.raises(ValueError, match=r"must return leading shape \(N_env, \.\.\.\)"):
        manager.compute(_make_observation_context(other_agent_x=0.8))


def test_simple_state_observation_config_outputs_goal_and_velocity_only() -> None:
    manager = ObservationManager(build_simple_state_observation_config().terms)

    obs = manager.compute(_make_observation_context(other_agent_x=0.8, controlled_velocity=(0.3, -0.1), goal_xy=(1.5, 0.2)))

    assert set(obs.keys()) == {"current_velocity", "goal_position"}
    assert torch.allclose(obs["current_velocity"], torch.tensor([[0.3, -0.1]], dtype=torch.float32))
    assert torch.allclose(obs["goal_position"], torch.tensor([[1.5, 0.2]], dtype=torch.float32))


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))