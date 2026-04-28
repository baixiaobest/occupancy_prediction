from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces


class MinimalKinematicsObservationWrapper(gym.ObservationWrapper):
    """Project dict observations to a minimal kinematic subset.

    Keeps only:
    - goal_position
    - current_velocity
    - last_commanded_velocity
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("MinimalKinematicsObservationWrapper requires Dict observation_space")

        required_keys = ("goal_position", "current_velocity", "last_commanded_velocity")
        missing = [key for key in required_keys if key not in self.observation_space.spaces]
        if missing:
            raise KeyError(f"Missing required observation keys for minimal mode: {missing}")

        self.observation_space = spaces.Dict(
            {
                "goal_position": self.observation_space.spaces["goal_position"],
                "current_velocity": self.observation_space.spaces["current_velocity"],
                "last_commanded_velocity": self.observation_space.spaces["last_commanded_velocity"],
            }
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "goal_position": observation["goal_position"],
            "current_velocity": observation["current_velocity"],
            "last_commanded_velocity": observation["last_commanded_velocity"],
        }


__all__ = ["MinimalKinematicsObservationWrapper"]