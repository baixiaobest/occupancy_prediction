from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import torch

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")
pytest.importorskip("rvo2")

from src.scene import AgentSpec, Scene
from src.skrl.env_torch_orca import TorchORCAEnv


def _make_scene() -> Scene:
    return Scene(
        agents=[
            AgentSpec(position=(0.0, 0.0), goal=(1.0, 0.0)),
            AgentSpec(position=(1.0, 1.0), goal=(2.0, 1.0)),
        ],
        obstacles=[],
    )


def _make_env() -> TorchORCAEnv:
    return TorchORCAEnv(scene_factory=_make_scene)


def test_step_clips_action_and_scales_velocity(monkeypatch) -> None:
    env = _make_env()

    class _DummySim:
        def __init__(self) -> None:
            self.captured: dict[int, np.ndarray] | None = None

        def step(self, *, controlled_pref_velocities, return_velocities):
            assert return_velocities is True
            self.captured = controlled_pref_velocities
            positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            velocities = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            return positions, velocities

    sim = _DummySim()
    env.sim = sim
    env._goals = torch.tensor([[1.0, 0.0], [2.0, 1.0]], dtype=torch.float32)
    env._last_positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    env._last_velocities = torch.zeros((2, 2), dtype=torch.float32)

    monkeypatch.setattr(env, "_build_obs", lambda positions, velocities: {})
    monkeypatch.setattr(
        env,
        "_compute_reward_terminated",
        lambda prev_positions, new_positions, new_velocities: (0.0, False, {"reward_terms": {}}),
    )

    env.step(torch.tensor([2.0, -2.0]))

    assert sim.captured is not None
    commanded = sim.captured[int(env.config.controlled_agent_index)]
    assert np.allclose(commanded, np.array([2.0, -2.0], dtype=np.float32))


def test_compute_progress_uses_goal_distance_delta() -> None:
    env = _make_env()
    env._goals = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    prev_positions = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    new_positions = torch.tensor([[0.5, 0.0], [0.0, 0.0]], dtype=torch.float32)

    prev_d, new_d, progress = env._compute_progress(prev_positions, new_positions)
    assert prev_d == pytest.approx(1.0)
    assert new_d == pytest.approx(0.5)
    assert progress == pytest.approx(0.5)


def test_compute_collision_detects_nearby_agent() -> None:
    env = _make_env()
    env.config.reward.collision_distance = 0.2
    near = torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
    far = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    assert env._compute_collision(near) is True
    assert env._compute_collision(far) is False


def test_compute_success_state_requires_within_goal_and_stationary() -> None:
    env = _make_env()
    env.sim = SimpleNamespace(goal_tolerance=0.2)
    velocities = torch.tensor([[0.05, 0.0], [0.0, 0.0]], dtype=torch.float32)

    speed, within_goal, stationary, success = env._compute_success_state(
        new_goal_distance=0.1,
        new_velocities=velocities,
    )

    assert speed == pytest.approx(0.05)
    assert within_goal is True
    assert stationary is True
    assert success is True


def test_action_change_penalty_uses_last_commanded_velocity() -> None:
    env = _make_env()
    env.config.reward.action_change_penalty_weight = 0.5
    env._last_commanded_velocity = torch.tensor([1.0, 0.0], dtype=torch.float32)

    change, penalty = env._compute_action_change_penalty(torch.tensor([0.0, 0.0], dtype=torch.float32))

    assert change == pytest.approx(1.0)
    assert penalty == pytest.approx(-0.5)


def test_compute_reward_terminated_triggers_when_goal_distance_too_far(monkeypatch) -> None:
    env = _make_env()
    env.sim = SimpleNamespace(goal_tolerance=0.2)
    env.config.reward.max_goal_distance_termination = 2.0

    prev_positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    new_positions = torch.tensor([[3.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    new_velocities = torch.zeros((2, 2), dtype=torch.float32)

    monkeypatch.setattr(env, "_compute_progress", lambda prev, new: (1.0, 3.0, -2.0))
    monkeypatch.setattr(env, "_compute_collision", lambda positions: False)
    monkeypatch.setattr(env, "_compute_success_state", lambda new_goal_distance, new_velocities: (0.0, False, True, False))

    reward, terminated, info = env._compute_reward_terminated(
        prev_positions=prev_positions,
        new_positions=new_positions,
        new_velocities=new_velocities,
    )

    assert isinstance(reward, float)
    assert terminated is True
    assert info["too_far"] is True
    assert info["distance_to_goal"] == pytest.approx(3.0)
    assert info["termination_reasons"] == ["too_far"]


def test_compute_reward_terminated_does_not_trigger_too_far_when_disabled(monkeypatch) -> None:
    env = _make_env()
    env.sim = SimpleNamespace(goal_tolerance=0.2)
    env.config.reward.max_goal_distance_termination = None

    prev_positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    new_positions = torch.tensor([[3.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    new_velocities = torch.zeros((2, 2), dtype=torch.float32)

    monkeypatch.setattr(env, "_compute_progress", lambda prev, new: (1.0, 3.0, -2.0))
    monkeypatch.setattr(env, "_compute_collision", lambda positions: False)
    monkeypatch.setattr(env, "_compute_success_state", lambda new_goal_distance, new_velocities: (0.0, False, True, False))

    _, terminated, info = env._compute_reward_terminated(
        prev_positions=prev_positions,
        new_positions=new_positions,
        new_velocities=new_velocities,
    )

    assert terminated is False
    assert info["too_far"] is False
    assert info["max_goal_distance_termination"] is None
    assert info["termination_reasons"] == []


def test_evaluate_termination_checks_terminates_when_any_check_is_true() -> None:
    env = _make_env()

    terminated, reasons = env._evaluate_termination_checks(
        {
            "success": False,
            "too_far": True,
            "collision": False,
            "goal_distance": 3.0,
            "step_count": 12,
        }
    )

    assert terminated is True
    assert reasons == ["too_far"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
