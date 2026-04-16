from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from src.rl.collector import (
    RandomPlanCollector,
    RandomPlanCollectorConfig,
)
from src.rl.observation_manager import (
    build_observation_manager,
    build_online_occupancy_observation_config,
    OnlineOccupancyObservationConfig,
)
from src.rl.replay_buffer import ReplayBuffer
from src.scene import AgentSpec, ObstacleSpec, Scene


class _DummyEnv:
    def __init__(self) -> None:
        self.dt = 0.1
        self._scene = Scene(
            agents=[
                AgentSpec(position=(0.0, 0.0), goal=(2.0, 0.0)),
                AgentSpec(position=(1.0, 0.0), goal=(1.0, 0.0)),
            ],
            obstacles=[
                ObstacleSpec(vertices=[(0.6, -0.4), (0.9, -0.4), (0.9, 0.4), (0.6, 0.4)]),
            ],
        )
        self.sim = SimpleNamespace(scene=self._scene)
        self._positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        self._velocities = torch.zeros((2, 2), dtype=torch.float32)
        self._goals = torch.tensor([[2.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        self._step_count = 0
        self.last_action = None

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        del seed
        self._positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        self._velocities = torch.zeros((2, 2), dtype=torch.float32)
        self._step_count = 0
        self.last_action = None
        return self._build_obs()

    def step(self, action_velocity: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, bool]]]:
        action = torch.as_tensor(action_velocity, dtype=torch.float32)
        self.last_action = action.clone()
        self._positions[0] = self._positions[0] + action * self.dt
        self._velocities[0] = action
        self._velocities[1] = torch.tensor([0.0, 0.1], dtype=torch.float32)
        self._positions[1] = self._positions[1] + self._velocities[1] * self.dt
        self._step_count += 1
        done = torch.tensor([self._step_count >= 2], dtype=torch.bool)
        reward = torch.tensor([1.5], dtype=torch.float32)
        return self._build_obs(), reward, done, [{"timeout": False}]

    def _build_obs(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self._positions.unsqueeze(0),
            "velocities": self._velocities.unsqueeze(0),
            "goals": self._goals.unsqueeze(0),
            "controlled_agent_index": torch.tensor([0], dtype=torch.int64),
            "step_count": torch.tensor([self._step_count], dtype=torch.int64),
        }


def test_random_plan_collector_adds_replay_transitions() -> None:
    env = _DummyEnv()
    replay_buffer = ReplayBuffer(capacity=16, seed=0)
    observation_config = build_online_occupancy_observation_config(
        OnlineOccupancyObservationConfig(
            decoder_context_len=4,
            local_map_shape=(12, 10),
            occupancy_resolution=(0.2, 0.2),
            agent_radius=0.2,
        )
    )
    observation_manager = build_observation_manager(observation_config)
    collector = RandomPlanCollector(
        env=env,
        replay_buffer=replay_buffer,
        observation_manager=observation_manager,
        config=RandomPlanCollectorConfig(
            horizon=5,
            num_candidates=4,
            max_speed=2.0,
            delta_std=0.3,
            dt=env.dt,
            action_selection="first",
            seed=7,
        ),
    )

    summary = collector.collect_steps(3)

    assert summary.transitions_added == 3
    assert summary.episodes_completed == 1
    assert summary.total_reward == pytest.approx(4.5)
    assert len(replay_buffer) == 3

    item = replay_buffer._items[0]
    assert tuple(item["obs"]["dynamic_context"].shape) == (1, 4, 12, 10)
    assert tuple(item["obs"]["static_map"].shape) == (1, 12, 10)
    assert tuple(item["obs"]["current_velocity"].shape) == (2,)
    assert tuple(item["obs"]["goal_position"].shape) == (2,)
    assert tuple(item["actions"].shape) == (5, 2)
    assert tuple(item["candidate_actions"].shape) == (4, 5, 2)
    assert item["obs"]["static_map"].sum().item() > 0.0
    assert torch.allclose(item["actions"], item["candidate_actions"][0])
    assert torch.allclose(env.last_action, replay_buffer._items[-1]["actions"][0])


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))