from __future__ import annotations

import importlib

import numpy as np

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")

import gymnasium as gym

from src.skrl.observation_wrappers import MinimalKinematicsObservationWrapper


class _DummyDictObsEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "goal_position": gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
                "current_velocity": gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
                "last_commanded_velocity": gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
                "extra": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        obs = {
            "goal_position": np.array([1.0, 2.0], dtype=np.float32),
            "current_velocity": np.array([0.1, -0.2], dtype=np.float32),
            "last_commanded_velocity": np.array([0.0, 0.0], dtype=np.float32),
            "extra": np.array([1.0], dtype=np.float32),
        }
        return obs, {}

    def step(self, action: np.ndarray):
        del action
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}


class _DummyBoxObsEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}


def test_minimal_wrapper_selects_required_keys() -> None:
    env = _DummyDictObsEnv()
    wrapper = MinimalKinematicsObservationWrapper(env)

    obs, _ = wrapper.reset()
    assert set(obs.keys()) == {"goal_position", "current_velocity", "last_commanded_velocity"}
    assert set(wrapper.observation_space.spaces.keys()) == {"goal_position", "current_velocity", "last_commanded_velocity"}


def test_minimal_wrapper_rejects_nondict_space() -> None:
    env = _DummyBoxObsEnv()
    with pytest.raises(TypeError, match="requires Dict observation_space"):
        MinimalKinematicsObservationWrapper(env)


def test_minimal_wrapper_rejects_missing_keys() -> None:
    env = _DummyDictObsEnv()
    del env.observation_space.spaces["last_commanded_velocity"]
    with pytest.raises(KeyError, match="Missing required observation keys"):
        MinimalKinematicsObservationWrapper(env)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
