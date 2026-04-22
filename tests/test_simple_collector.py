from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.rl.collector.simple_collector as simple_collector_module
from src.rl import build_simple_state_observation_config
from src.rl.replay_buffer import ReplayBuffer
from src.rl.collector.simple_collector import (
    SimpleQActionSelectionConfig,
    SimpleRandomActionCollector,
    SimpleRandomActionCollectorConfig,
)
from src.scene import AgentSpec, Scene


class _DummyEnv:
    def __init__(self) -> None:
        self.dt = 0.1
        self._scene = Scene(
            agents=[
                AgentSpec(position=(0.0, 0.0), goal=(2.0, 0.0)),
                AgentSpec(position=(1.0, 0.0), goal=(1.0, 0.0)),
            ],
            obstacles=[],
        )
        self.sim = SimpleNamespace(scene=self._scene)
        self.env_config = SimpleNamespace(observation=build_simple_state_observation_config())
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

    def step(self, action_velocity: torch.Tensor):
        action = torch.as_tensor(action_velocity, dtype=torch.float32)
        self.last_action = action.clone()
        self._positions[0] = self._positions[0] + action * self.dt
        self._velocities[0] = action
        self._velocities[1] = torch.tensor([0.0, 0.1], dtype=torch.float32)
        self._positions[1] = self._positions[1] + self._velocities[1] * self.dt
        self._step_count += 1
        done = torch.tensor([self._step_count >= 2], dtype=torch.bool)
        reward = torch.tensor([0.75], dtype=torch.float32)
        return self._build_obs(), reward, done, [{"timeout": False}]

    def _build_obs(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self._positions.unsqueeze(0),
            "velocities": self._velocities.unsqueeze(0),
            "goals": self._goals.unsqueeze(0),
            "controlled_agent_index": torch.tensor([0], dtype=torch.int64),
        }


class _ActionXQNetwork(nn.Module):
    def forward(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_position: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        del current_velocity, goal_position
        return torch.as_tensor(action, dtype=torch.float32)[:, 0]


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
    del current_velocity, max_speed, delta_std, dt, include_current_velocity_candidate, generator
    plans = torch.zeros((1, num_candidates, horizon, 2), dtype=torch.float32)
    for candidate_idx in range(num_candidates):
        plans[0, candidate_idx, :, 0] = float(candidate_idx)
    return plans


def test_simple_random_action_collector_adds_replay_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simple_collector_module, "sample_random_velocity_plans", _dummy_candidate_sampler)

    env = _DummyEnv()
    replay_buffer = ReplayBuffer(capacity=16, seed=0)
    collector = SimpleRandomActionCollector(
        env=env,
        replay_buffer=replay_buffer,
        observation_manager=None,
        config=SimpleRandomActionCollectorConfig(
            num_candidates=4,
            max_speed=2.0,
            delta_std=0.0,
            dt=env.dt,
            action_selection="first",
            seed=7,
        ),
    )

    summary = collector.collect_steps(3)

    assert collector.observation_manager is not None
    assert summary.transitions_added == 3
    assert summary.episodes_completed == 1
    assert summary.total_reward == pytest.approx(2.25)
    assert len(replay_buffer) == 3

    item = replay_buffer._items[0]
    assert set(item["obs"].keys()) == {"current_velocity", "goal_position"}
    assert tuple(item["obs"]["current_velocity"].shape) == (2,)
    assert tuple(item["obs"]["goal_position"].shape) == (2,)
    assert tuple(item["actions"].shape) == (2,)
    assert tuple(item["candidate_actions"].shape) == (4, 2)
    assert torch.allclose(item["actions"], item["candidate_actions"][0])
    assert torch.allclose(env.last_action, replay_buffer._items[-1]["actions"])


def test_simple_random_action_collector_q_softmax_uses_q_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simple_collector_module, "sample_random_velocity_plans", _dummy_candidate_sampler)

    env = _DummyEnv()
    replay_buffer = ReplayBuffer(capacity=8, seed=0)
    collector = SimpleRandomActionCollector(
        env=env,
        replay_buffer=replay_buffer,
        observation_manager=None,
        config=SimpleRandomActionCollectorConfig(
            num_candidates=3,
            max_speed=2.0,
            delta_std=0.0,
            dt=env.dt,
            action_selection="q_softmax",
            seed=5,
            q_selection=SimpleQActionSelectionConfig(
                temperature=0.01,
                seed=0,
            ),
        ),
        q_network=_ActionXQNetwork(),
    )

    summary = collector.collect_steps(1)

    assert summary.transitions_added == 1
    assert len(replay_buffer) == 1

    item = replay_buffer._items[0]
    assert torch.allclose(item["actions"], item["candidate_actions"][2])
    assert torch.allclose(env.last_action, torch.tensor([2.0, 0.0], dtype=torch.float32))
    assert item["candidate_log_probs"] is not None
    candidate_probs = torch.exp(item["candidate_log_probs"])
    assert candidate_probs.argmax().item() == 2
    assert candidate_probs[2].item() > 0.999


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))