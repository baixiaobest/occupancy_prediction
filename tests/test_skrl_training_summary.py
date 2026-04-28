from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")

import gymnasium as gym
import numpy as np

from src.skrl.training_summary import (
    PeriodicEpisodeSummaryWrapper,
    _mean_metric,
    install_agent_tracking_summary,
)


class _DummyAgent:
    def __init__(self) -> None:
        self.tracking_data: dict[str, list[float]] = {}
        self.calls: list[tuple[int, int]] = []

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        self.calls.append((int(timestep), int(timesteps)))


class _DoneEveryStepEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._step_in_episode = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        self._step_in_episode = 0
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        self._step_in_episode += 1
        terminated = self._step_in_episode >= 2
        info = {
            "success": terminated,
            "collision": False,
            "timeout": False,
            "reward_terms": {
                "progress": 1.0,
                "step_penalty": -0.1,
                "action_change": -0.05,
            },
        }
        return np.zeros((2,), dtype=np.float32), 1.0, bool(terminated), False, info


def test_mean_metric_handles_missing_or_empty() -> None:
    assert _mean_metric({}, "x") is None
    assert _mean_metric({"x": []}, "x") is None


def test_mean_metric_returns_mean() -> None:
    value = _mean_metric({"x": [1.0, 2.0, 3.0]}, "x")
    assert value == pytest.approx(2.0)


def test_install_agent_tracking_summary_prints_and_calls_original(capsys) -> None:
    agent = _DummyAgent()
    agent.tracking_data = {
        "Loss / Policy loss": [0.2],
        "Loss / Value loss": [0.1],
        "Policy / Standard deviation": [0.6],
    }

    install_agent_tracking_summary(agent, prefix="[unit]")
    agent.write_tracking_data(32, 128)

    captured = capsys.readouterr().out
    assert "[unit] ppo" in captured
    assert "t=32/128" in captured
    assert "pi_loss=" in captured
    assert "vf_loss=" in captured
    assert "act_std=" in captured
    assert agent.calls == [(32, 128)]


def test_install_agent_tracking_summary_uses_last_seen_metrics(capsys) -> None:
    agent = _DummyAgent()
    install_agent_tracking_summary(agent, prefix="[unit]")

    agent.tracking_data = {"Loss / Policy loss": [0.5]}
    agent.write_tracking_data(1, 10)

    agent.tracking_data = {}
    agent.write_tracking_data(2, 10)

    captured = capsys.readouterr().out
    assert "t=1/10" in captured
    assert "pi_loss=0.50000" in captured
    assert "t=2/10" in captured
    assert "pi_loss=0.50000" in captured


def test_periodic_summary_wrapper_validates_interval() -> None:
    with pytest.raises(ValueError, match="interval_episodes must be > 0"):
        PeriodicEpisodeSummaryWrapper(_DoneEveryStepEnv(), interval_episodes=0)


def test_periodic_summary_wrapper_prints_summary(capsys) -> None:
    wrapped = PeriodicEpisodeSummaryWrapper(_DoneEveryStepEnv(), interval_episodes=2, prefix="[unit]")
    wrapped.reset()
    wrapped.step(np.zeros((2,), dtype=np.float32))
    wrapped.step(np.zeros((2,), dtype=np.float32))
    wrapped.reset()
    wrapped.step(np.zeros((2,), dtype=np.float32))
    wrapped.step(np.zeros((2,), dtype=np.float32))

    captured = capsys.readouterr().out
    assert "[unit] summary" in captured
    assert "mean_return=2.0000" in captured
    assert "success_rate=1.000" in captured
    assert "reward_terms_mean_per_episode\n" in captured
    assert "action_change=-0.1000\n" in captured
    assert "progress=2.0000" in captured
    assert "step_penalty=-0.2000\n" in captured


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
