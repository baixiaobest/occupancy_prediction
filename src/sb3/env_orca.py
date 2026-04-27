from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.ORCASim import ORCASim
from src.occupancy2d import Occupancy2d
from src.occupancy_patch import slice_centered_patch
from src.rollout_helpers import _build_scene_static_map, _compute_scene_canvas
from src.scene import Scene


@dataclass
class ORCASB3SimConfig:
    """Configuration forwarded to ORCASim for SB3 training."""

    time_step: float = 0.1
    neighbor_dist: float = 3.0
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
    collision_distance: float = 0.2
    success_speed_threshold: float = 0.1
    action_change_penalty_weight: float = 0.005


@dataclass
class ORCASB3OccupancyConfig:
    """Occupancy observation settings for SB3 policy inputs."""

    resolution: float = 0.1
    patch_length: float = 12.8
    patch_width: float = 12.8
    dynamic_context_len: int = 1
    agent_radius: float = 0.3


@dataclass
class ORCASB3EnvConfig:
    """Environment settings for the SB3 Gym wrapper."""

    max_steps: int = 200
    controlled_agent_index: int = 0
    controlled_agent_max_speed: float = 2.0
    sim: ORCASB3SimConfig = field(default_factory=ORCASB3SimConfig)
    reward: ORCASB3RewardConfig = field(default_factory=ORCASB3RewardConfig)
    occupancy: ORCASB3OccupancyConfig = field(default_factory=ORCASB3OccupancyConfig)


class ORCASB3Env(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Minimal Gymnasium environment around ORCASim for PPO.

    Observation keys:
    - dynamic_context: (1, T_ctx, H, W) dynamic local occupancy context.
    - static_map: (1, H, W) static local occupancy map.
    - goal_position: (2,) relative goal offset.
    - current_velocity: (2,) current controlled-agent velocity.
    - last_commanded_velocity: (2,) previous action mapped to velocity.

    Action: normalized command [ax, ay] in [-1, 1], scaled to m/s in step()
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
        self._controlled_agent_max_speed = float(self.config.controlled_agent_max_speed)
        if self._controlled_agent_max_speed <= 0.0:
            raise ValueError("controlled_agent_max_speed must be positive")
        if float(self.config.occupancy.resolution) <= 0.0:
            raise ValueError("occupancy.resolution must be positive")
        if float(self.config.occupancy.patch_length) <= 0.0:
            raise ValueError("occupancy.patch_length must be positive")
        if float(self.config.occupancy.patch_width) <= 0.0:
            raise ValueError("occupancy.patch_width must be positive")
        if int(self.config.occupancy.dynamic_context_len) <= 0:
            raise ValueError("occupancy.dynamic_context_len must be > 0")
        if float(self.config.occupancy.agent_radius) < 0.0:
            raise ValueError("occupancy.agent_radius must be >= 0")

        occ_cfg = self.config.occupancy
        self._occupancy_resolution = float(occ_cfg.resolution)
        self._occupancy_resolution_xy = (self._occupancy_resolution, self._occupancy_resolution)
        self._local_map_shape = (
            int(np.floor(float(occ_cfg.patch_width) / self._occupancy_resolution)),
            int(np.floor(float(occ_cfg.patch_length) / self._occupancy_resolution)),
        )
        if self._local_map_shape[0] <= 0 or self._local_map_shape[1] <= 0:
            raise ValueError("occupancy patch size/resolution yields non-positive local map shape")
        self._dynamic_context_len = int(occ_cfg.dynamic_context_len)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "dynamic_context": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self._dynamic_context_len, self._local_map_shape[0], self._local_map_shape[1]),
                    dtype=np.float32,
                ),
                "static_map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self._local_map_shape[0], self._local_map_shape[1]),
                    dtype=np.float32,
                ),
                "goal_position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "current_velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "last_commanded_velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self.sim: ORCASim | None = None
        self._goals: np.ndarray | None = None
        self._last_positions: np.ndarray | None = None
        self._last_velocities: np.ndarray | None = None
        self._last_commanded_velocity: np.ndarray | None = None
        self._scene_static_map: torch.Tensor | None = None
        self._scene_origin: tuple[float, float] | None = None
        self._scene_center: np.ndarray | None = None
        self._scene_canvas_shape: tuple[int, int] | None = None
        self._dynamic_renderer: Occupancy2d | None = None
        self._dynamic_context_history: deque[torch.Tensor] = deque(maxlen=self._dynamic_context_len + 1)
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        self._last_commanded_velocity = None
        self._initialize_occupancy(scene=self.sim.scene, positions=positions)

        obs = self._build_obs(positions, velocities)
        info = {
            "goal_distance": float(self._goal_distance(positions)),
            "step_count": int(self._step_count),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.sim is None or self._last_positions is None or self._last_velocities is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        commanded_velocity = action * self._controlled_agent_max_speed
        controlled_idx = int(self.config.controlled_agent_index)

        new_positions, new_velocities = self.sim.step(
            controlled_pref_velocities={controlled_idx: commanded_velocity},
            return_velocities=True,
        )
        new_positions = np.asarray(new_positions, dtype=np.float32)
        new_velocities = np.asarray(new_velocities, dtype=np.float32)

        reward, terminated, info = self._compute_reward_terminated(
            prev_positions=self._last_positions,
            new_positions=new_positions,
            new_velocities=new_velocities,
        )

        action_change, action_change_penalty = self._compute_action_change_penalty(commanded_velocity)
        reward += action_change_penalty

        info["action_change"] = float(action_change)
        info["reward_terms"]["action_change"] = float(action_change_penalty)

        self._last_positions = new_positions
        self._last_velocities = new_velocities
        self._last_commanded_velocity = commanded_velocity.copy()
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

    def _initialize_occupancy(self, *, scene: Scene, positions: np.ndarray) -> None:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")

        traj_seed = np.stack([positions, self._goals], axis=0)
        scene_origin, scene_size, scene_canvas_shape, scene_center = _compute_scene_canvas(
            traj=traj_seed,
            obstacles=scene.obstacles,
            resolution=self._occupancy_resolution,
            occupancy_length=float(self.config.occupancy.patch_length),
            occupancy_width=float(self.config.occupancy.patch_width),
        )
        self._scene_origin = (float(scene_origin[0]), float(scene_origin[1]))
        self._scene_center = np.asarray(scene_center, dtype=np.float32)
        self._scene_canvas_shape = (int(scene_canvas_shape[0]), int(scene_canvas_shape[1]))
        self._scene_static_map = _build_scene_static_map(
            obstacles=scene.obstacles,
            resolution=self._occupancy_resolution,
            agent_radius=float(self.config.occupancy.agent_radius),
            scene_size=scene_size,
            scene_center=self._scene_center,
        ).to(dtype=torch.float32)
        self._dynamic_renderer = Occupancy2d(
            resolution=self._occupancy_resolution_xy,
            size=scene_size,
            trajectory=None,
            static_obstacles=None,
            agent_radius=float(self.config.occupancy.agent_radius),
        )
        self._dynamic_context_history.clear()

    def _build_obs(self, positions: np.ndarray, velocities: np.ndarray) -> dict[str, np.ndarray]:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")
        if self._scene_static_map is None or self._scene_origin is None:
            raise RuntimeError("Occupancy state is not initialized")
        if self._scene_center is None or self._scene_canvas_shape is None or self._dynamic_renderer is None:
            raise RuntimeError("Dynamic occupancy state is not initialized")

        controlled_idx = int(self.config.controlled_agent_index)
        goal_offset = self._goals[controlled_idx] - positions[controlled_idx]
        current_velocity = velocities[controlled_idx]
        if self._last_commanded_velocity is None:
            last_commanded_velocity = np.zeros(2, dtype=np.float32)
        else:
            last_commanded_velocity = self._last_commanded_velocity

        center_xy = torch.as_tensor(positions[controlled_idx], dtype=torch.float32)
        static_local = slice_centered_patch(
            self._scene_static_map,
            center_xy,
            self._scene_origin,
            self._occupancy_resolution_xy,
            self._local_map_shape,
            binary=False,
            prefer_view=True,
        )

        num_agents = int(positions.shape[0])
        other_indices = [idx for idx in range(num_agents) if idx != controlled_idx]
        if other_indices:
            other_positions = positions[other_indices].astype(np.float32, copy=False)[None, :, :]
            self._dynamic_renderer.update_inputs(trajectory=other_positions, static_obstacles=[])
            dynamic_global = self._dynamic_renderer.generate(center_offset=tuple(self._scene_center.tolist()))[0]
            dynamic_global = dynamic_global.to(dtype=torch.float32)
        else:
            dynamic_global = torch.zeros(self._scene_canvas_shape, dtype=torch.float32)

        dynamic_local = slice_centered_patch(
            dynamic_global,
            center_xy,
            self._scene_origin,
            self._occupancy_resolution_xy,
            self._local_map_shape,
            binary=False,
            prefer_view=True,
        )

        self._dynamic_context_history.append(dynamic_local)
        context_frames = list(self._dynamic_context_history)[:-1]
        if not context_frames:
            context_frames = [dynamic_local]
        context_frames = context_frames[-self._dynamic_context_len :]
        while len(context_frames) < self._dynamic_context_len:
            context_frames.insert(0, context_frames[0])
        dynamic_context = torch.stack(context_frames, dim=0).unsqueeze(0)

        obs = {
            "dynamic_context": dynamic_context.cpu().numpy().astype(np.float32, copy=False),
            "static_map": static_local.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False),
            "goal_position": np.asarray(goal_offset, dtype=np.float32),
            "current_velocity": np.asarray(current_velocity, dtype=np.float32),
            "last_commanded_velocity": np.asarray(last_commanded_velocity, dtype=np.float32),
        }
        return obs

    def _compute_reward_terminated(
        self,
        *,
        prev_positions: np.ndarray,
        new_positions: np.ndarray,
        new_velocities: np.ndarray,
    ) -> tuple[float, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        prev_goal_distance, new_goal_distance, progress = self._compute_progress(prev_positions, new_positions)
        collision = self._compute_collision(new_positions)
        controlled_speed, within_goal, stationary, success = self._compute_success_state(
            new_goal_distance=new_goal_distance,
            new_velocities=new_velocities,
        )

        reward_terms = {
            "progress": self._compute_progress_reward(progress),
            "step_penalty": self._compute_step_penalty_reward(),
            "collision": self._compute_collision_reward(collision),
            "success": self._compute_success_reward(success),
        }
        reward = float(sum(reward_terms.values()))

        info = {
            "success": success,
            "collision": collision,
            "timeout": False,
            "goal_distance": float(new_goal_distance),
            "controlled_speed": float(controlled_speed),
            "within_goal": bool(within_goal),
            "stationary": bool(stationary),
            "success_speed_threshold": float(self.config.reward.success_speed_threshold),
            "progress": float(progress),
            "reward_terms": reward_terms,
        }
        terminated = success
        return float(reward), bool(terminated), info

    def _compute_progress(self, prev_positions: np.ndarray, new_positions: np.ndarray) -> tuple[float, float, float]:
        prev_goal_distance = self._goal_distance(prev_positions)
        new_goal_distance = self._goal_distance(new_positions)
        progress = float(prev_goal_distance - new_goal_distance)
        return float(prev_goal_distance), float(new_goal_distance), float(progress)

    def _compute_collision(self, new_positions: np.ndarray) -> bool:
        controlled_idx = int(self.config.controlled_agent_index)
        controlled_pos = new_positions[controlled_idx]
        diffs = new_positions - controlled_pos[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        self_mask = np.zeros_like(dists, dtype=bool)
        self_mask[controlled_idx] = True
        return bool(np.any((dists < float(self.config.reward.collision_distance)) & (~self_mask)))

    def _compute_success_state(
        self,
        *,
        new_goal_distance: float,
        new_velocities: np.ndarray,
    ) -> tuple[float, bool, bool, bool]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        controlled_idx = int(self.config.controlled_agent_index)
        controlled_speed = float(np.linalg.norm(new_velocities[controlled_idx]))
        within_goal = bool(new_goal_distance <= float(self.sim.goal_tolerance))
        stationary = bool(controlled_speed <= float(self.config.reward.success_speed_threshold))
        success = bool(within_goal and stationary)
        return float(controlled_speed), bool(within_goal), bool(stationary), bool(success)

    def _compute_progress_reward(self, progress: float) -> float:
        return float(self.config.reward.progress_weight) * float(progress)

    def _compute_step_penalty_reward(self) -> float:
        return float(self.config.reward.step_penalty)

    def _compute_collision_reward(self, collision: bool) -> float:
        return float(self.config.reward.collision_penalty) if collision else 0.0

    def _compute_success_reward(self, success: bool) -> float:
        return float(self.config.reward.success_reward) if success else 0.0

    def _compute_action_change_penalty(self, commanded_velocity: np.ndarray) -> tuple[float, float]:
        action_change = 0.0
        if self._last_commanded_velocity is not None:
            action_change = float(np.linalg.norm(commanded_velocity - self._last_commanded_velocity))
        penalty = -float(self.config.reward.action_change_penalty_weight) * float(action_change)
        return float(action_change), float(penalty)


__all__ = [
    "ORCASB3Env",
    "ORCASB3EnvConfig",
    "ORCASB3OccupancyConfig",
    "ORCASB3RewardConfig",
    "ORCASB3SimConfig",
]
