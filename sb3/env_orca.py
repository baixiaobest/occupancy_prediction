from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ORCASim import ORCASim
from src.scene import Scene


@dataclass
class ORCASB3SimConfig:
    """Configuration forwarded to ORCASim for SB3 training."""

    time_step: float = 0.1
    neighbor_dist: float = 1.0
    max_neighbors: int = 5
    time_horizon: float = 3.0
    time_horizon_obst: float = 5.0
    radius: float = 0.3
    max_speed: float = 3.0
    goal_tolerance: float = 0.2
    path_goal_switch_tolerance: float = 3.0
    path_segment_remaining_switch_ratio: float = 0.05
    pref_velocity_noise_std: float = 0.02
    pref_velocity_noise_interval: int = 3
    pref_velocity_noise_seed: int = 0
    lateral_control_gain: float = 1.0
    lateral_control_max_speed: float = 1.0
    strict_controlled_agent: bool = True
    strict_control_velocity_tolerance: float = 1e-3
    strict_control_assert: bool = False


@dataclass
class ORCASB3RewardConfig:
    """Simple reward terms matching the current RL setup."""

    progress_weight: float = 1.0
    step_penalty: float = 0.0
    collision_penalty: float = -1.0
    success_reward: float = 5.0
    collision_distance: float = 0.4


@dataclass
class ORCASB3EnvConfig:
    """Environment settings for the SB3 Gym wrapper."""

    max_steps: int = 200
    controlled_agent_index: int = 0
    sim: ORCASB3SimConfig = field(default_factory=ORCASB3SimConfig)
    reward: ORCASB3RewardConfig = field(default_factory=ORCASB3RewardConfig)


class ORCASB3Env(gym.Env[np.ndarray, np.ndarray]):
    """Minimal Gymnasium environment around ORCASim for PPO.

    Observation: [relative_goal_x, relative_goal_y, vel_x, vel_y]
    Action: commanded preferred velocity [vx, vy] for the controlled agent
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scene_factory: Callable[[], Scene],
        config: ORCASB3EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.scene_factory = scene_factory
        self.config = config if config is not None else ORCASB3EnvConfig()

        max_speed = float(self.config.sim.max_speed)
        self.action_space = spaces.Box(
            low=np.array([-max_speed, -max_speed], dtype=np.float32),
            high=np.array([max_speed, max_speed], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.sim: ORCASim | None = None
        self._goals: np.ndarray | None = None
        self._last_positions: np.ndarray | None = None
        self._last_velocities: np.ndarray | None = None
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)

        scene = self.scene_factory()
        sim_cfg = self.config.sim

        kwargs: dict[str, Any] = {
            "time_step": sim_cfg.time_step,
            "neighbor_dist": sim_cfg.neighbor_dist,
            "max_neighbors": sim_cfg.max_neighbors,
            "time_horizon": sim_cfg.time_horizon,
            "time_horizon_obst": sim_cfg.time_horizon_obst,
            "radius": sim_cfg.radius,
            "max_speed": sim_cfg.max_speed,
            "goal_tolerance": sim_cfg.goal_tolerance,
            "path_goal_switch_tolerance": sim_cfg.path_goal_switch_tolerance,
            "path_segment_remaining_switch_ratio": sim_cfg.path_segment_remaining_switch_ratio,
            "pref_velocity_noise_std": sim_cfg.pref_velocity_noise_std,
            "pref_velocity_noise_interval": sim_cfg.pref_velocity_noise_interval,
            "pref_velocity_noise_seed": sim_cfg.pref_velocity_noise_seed,
            "lateral_control_gain": sim_cfg.lateral_control_gain,
            "lateral_control_max_speed": sim_cfg.lateral_control_max_speed,
        }

        if sim_cfg.strict_controlled_agent:
            kwargs["strict_controlled_agent_index"] = int(self.config.controlled_agent_index)
            kwargs["strict_control_velocity_tolerance"] = float(sim_cfg.strict_control_velocity_tolerance)
            kwargs["strict_control_assert"] = bool(sim_cfg.strict_control_assert)
        if seed is not None:
            kwargs["region_pair_seed"] = int(seed)

        self.sim = ORCASim(scene=scene, **kwargs)
        self._step_count = 0

        positions = np.asarray(self.sim.get_agent_positions(), dtype=np.float32)
        velocities = np.asarray(self.sim.get_agent_velocities(), dtype=np.float32)
        self._validate_controlled_agent_index(num_agents=int(positions.shape[0]))

        self._goals = np.asarray([agent.goal for agent in self.sim.scene.agents], dtype=np.float32)
        self._last_positions = positions
        self._last_velocities = velocities

        obs = self._build_obs(positions, velocities)
        info = {
            "goal_distance": float(self._goal_distance(positions)),
            "step_count": int(self._step_count),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None or self._last_positions is None or self._last_velocities is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        controlled_idx = int(self.config.controlled_agent_index)

        new_positions, new_velocities = self.sim.step(
            controlled_pref_velocities={controlled_idx: action},
            return_velocities=True,
        )
        new_positions = np.asarray(new_positions, dtype=np.float32)
        new_velocities = np.asarray(new_velocities, dtype=np.float32)

        reward, terminated, info = self._compute_reward_terminated(
            prev_positions=self._last_positions,
            new_positions=new_positions,
        )

        self._last_positions = new_positions
        self._last_velocities = new_velocities
        self._step_count += 1

        truncated = self._step_count >= int(self.config.max_steps)
        if truncated:
            info["timeout"] = True

        obs = self._build_obs(new_positions, new_velocities)
        info["step_count"] = int(self._step_count)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _validate_controlled_agent_index(self, *, num_agents: int) -> None:
        idx = int(self.config.controlled_agent_index)
        if idx < 0 or idx >= num_agents:
            raise ValueError(f"controlled_agent_index {idx} is out of range for {num_agents} agents")

    def _goal_distance(self, positions: np.ndarray) -> float:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")
        controlled_idx = int(self.config.controlled_agent_index)
        diff = self._goals[controlled_idx] - positions[controlled_idx]
        return float(np.linalg.norm(diff))

    def _build_obs(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")

        controlled_idx = int(self.config.controlled_agent_index)
        goal_offset = self._goals[controlled_idx] - positions[controlled_idx]
        current_velocity = velocities[controlled_idx]

        obs = np.array(
            [goal_offset[0], goal_offset[1], current_velocity[0], current_velocity[1]],
            dtype=np.float32,
        )
        return obs

    def _compute_reward_terminated(
        self,
        *,
        prev_positions: np.ndarray,
        new_positions: np.ndarray,
    ) -> tuple[float, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        reward_cfg = self.config.reward
        controlled_idx = int(self.config.controlled_agent_index)

        prev_goal_distance = self._goal_distance(prev_positions)
        new_goal_distance = self._goal_distance(new_positions)
        progress = prev_goal_distance - new_goal_distance

        controlled_pos = new_positions[controlled_idx]
        diffs = new_positions - controlled_pos[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        self_mask = np.zeros_like(dists, dtype=bool)
        self_mask[controlled_idx] = True
        collision = bool(np.any((dists < float(reward_cfg.collision_distance)) & (~self_mask)))

        success = bool(new_goal_distance <= float(self.sim.goal_tolerance))

        reward = 0.0
        reward += float(reward_cfg.progress_weight) * float(progress)
        reward += float(reward_cfg.step_penalty)
        if collision:
            reward += float(reward_cfg.collision_penalty)
        if success:
            reward += float(reward_cfg.success_reward)

        info = {
            "success": success,
            "collision": collision,
            "timeout": False,
            "goal_distance": float(new_goal_distance),
            "progress": float(progress),
            "reward_terms": {
                "progress": float(reward_cfg.progress_weight) * float(progress),
                "step_penalty": float(reward_cfg.step_penalty),
                "collision": float(reward_cfg.collision_penalty) if collision else 0.0,
                "success": float(reward_cfg.success_reward) if success else 0.0,
            },
        }
        terminated = success
        return float(reward), bool(terminated), info


__all__ = [
    "ORCASB3Env",
    "ORCASB3EnvConfig",
    "ORCASB3RewardConfig",
    "ORCASB3SimConfig",
]
