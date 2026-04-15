from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("rvo2")

from src.ORCASim import ORCASim
from src.scene import AgentSpec, Scene


def _make_two_agent_head_on_scene() -> Scene:
    return Scene(
        agents=[
            AgentSpec(position=(0.0, 0.0), goal=(5.0, 0.0)),
            AgentSpec(position=(0.8, 0.0), goal=(-5.0, 0.0)),
        ],
        obstacles=[],
    )


def _supports_strict_control(sim: ORCASim) -> bool:
    required = (
        "setAgentMaxNeighbors",
        "setAgentNeighborDist",
        "setAgentTimeHorizon",
        "setAgentTimeHorizonObst",
        "setAgentMaxSpeed",
    )
    return all(hasattr(sim.sim, name) for name in required)


def _pairwise_distance(positions: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float(np.linalg.norm(positions[idx_a] - positions[idx_b]))


def test_strict_controlled_agent_tracks_commanded_velocity() -> None:
    sim = ORCASim(
        scene=_make_two_agent_head_on_scene(),
        time_step=0.1,
        neighbor_dist=5.0,
        max_neighbors=10,
        time_horizon=5.0,
        time_horizon_obst=5.0,
        radius=0.3,
        max_speed=1.5,
        pref_velocity_noise_std=0.0,
        lateral_control_gain=0.0,
        strict_controlled_agent_index=0,
        strict_control_velocity_tolerance=1e-4,
        strict_control_assert=True,
    )
    if not _supports_strict_control(sim):
        pytest.skip("python-rvo2 binding does not expose required per-agent strict-control setters")

    commanded_velocity = np.array([2.0, 0.0], dtype=np.float32)
    positions, velocities = sim.step(
        controlled_pref_velocities={0: commanded_velocity},
        return_velocities=True,
    )

    assert positions.shape == (2, 2)
    assert velocities.shape == (2, 2)
    assert np.allclose(velocities[0], commanded_velocity, atol=1e-4)


def test_strict_controlled_agent_can_be_commanded_into_collision() -> None:
    radius = 0.3
    sim = ORCASim(
        scene=_make_two_agent_head_on_scene(),
        time_step=0.1,
        neighbor_dist=5.0,
        max_neighbors=10,
        time_horizon=5.0,
        time_horizon_obst=5.0,
        radius=radius,
        max_speed=1.5,
        pref_velocity_noise_std=0.0,
        lateral_control_gain=0.0,
        strict_controlled_agent_index=0,
        strict_control_velocity_tolerance=1e-4,
        strict_control_assert=True,
    )
    if not _supports_strict_control(sim):
        pytest.skip("python-rvo2 binding does not expose required per-agent strict-control setters")

    commanded_velocity = np.array([2.0, 0.0], dtype=np.float32)
    collision_distance = 2.0 * radius
    collided = False
    min_distance = math.inf

    for _ in range(10):
        positions, velocities = sim.step(
            controlled_pref_velocities={0: commanded_velocity},
            return_velocities=True,
        )
        distance = _pairwise_distance(positions, 0, 1)
        min_distance = min(min_distance, distance)

        assert np.allclose(velocities[0], commanded_velocity, atol=1e-4)
        if distance < collision_distance:
            collided = True
            break

    assert collided, f"Expected commanded collision, but min distance was {min_distance:.4f}"


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
