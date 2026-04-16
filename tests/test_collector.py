from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.rl.collector import (
    QActionSelectionConfig,
    RandomPlanCollector,
    RandomPlanCollectorConfig,
)
from src.rl.counterfactual import CounterfactualRolloutBatch
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


class _DummyQNetwork(nn.Module):
    def forward(
        self,
        goal_position: torch.Tensor,
        current_velocity: torch.Tensor,
        planned_velocities: torch.Tensor,
        tapped_future_features: torch.Tensor,
    ) -> torch.Tensor:
        del goal_position, current_velocity
        taps = torch.as_tensor(tapped_future_features, dtype=torch.float32).reshape(planned_velocities.shape[0], -1)
        return taps.mean(dim=1)


class _DummyDecoder(nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("collector test dummy decoder should not be called directly")


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


def test_random_plan_collector_q_softmax_selection_uses_q_scores() -> None:
    env = _DummyEnv()
    replay_buffer = ReplayBuffer(capacity=8, seed=0)
    observation_config = build_online_occupancy_observation_config(
        OnlineOccupancyObservationConfig(
            decoder_context_len=2,
            local_map_shape=(8, 8),
            occupancy_resolution=(0.2, 0.2),
            agent_radius=0.2,
        )
    )
    collector = RandomPlanCollector(
        env=env,
        replay_buffer=replay_buffer,
        observation_manager=build_observation_manager(observation_config),
        config=RandomPlanCollectorConfig(
            horizon=3,
            num_candidates=3,
            max_speed=2.0,
            delta_std=0.0,
            dt=env.dt,
            action_selection="q_softmax",
            seed=5,
            q_selection=QActionSelectionConfig(
                temperature=0.1,
                seed=0,
                tap_layer=1,
                latent_channels=1,
                latent_shape=(1, 1, 1),
            ),
        ),
        q_network=_DummyQNetwork(),
        decoder=_DummyDecoder(),
        candidate_sampler=_dummy_candidate_sampler,
        counterfactual_rollout_fn=_dummy_rollout_fn,
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