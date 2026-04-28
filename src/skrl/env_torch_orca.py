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
from src.rollout_helpers import _compute_scene_canvas
from src.scene import Scene
import math


@dataclass
class TorchORCASimConfig:
    """Configuration forwarded to ORCASim."""

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
class TorchORCARewardConfig:
    """Reward terms matching ORCASB3Env behavior."""

    progress_weight: float = 1.0
    step_penalty: float = -0.01
    collision_penalty: float = -1.0
    success_reward: float = 20.0
    collision_distance: float = 0.2
    success_speed_threshold: float = 0.1
    success_distance: float = 0.3
    action_change_penalty_weight: float = 0.05
    max_goal_distance_termination: float | None = 12.0


@dataclass
class TorchORCAOccupancyConfig:
    """Occupancy observation settings."""

    resolution: float = 0.1
    patch_length: float = 12.8
    patch_width: float = 12.8
    dynamic_context_len: int = 1
    agent_radius: float = 0.3


@dataclass
class TorchORCAEnvConfig:
    """Environment settings for torch-first ORCA env."""

    max_steps: int = 200
    controlled_agent_index: int = 0
    controlled_agent_max_speed: float = 2.0
    device: str = "cpu"
    sim: TorchORCASimConfig = field(default_factory=TorchORCASimConfig)
    reward: TorchORCARewardConfig = field(default_factory=TorchORCARewardConfig)
    occupancy: TorchORCAOccupancyConfig = field(default_factory=TorchORCAOccupancyConfig)


class TorchORCAEnv(gym.Env[dict[str, torch.Tensor], torch.Tensor]):
    """Torch-internal ORCA environment for SKRL integration.

    Behavior mirrors ORCASB3Env while keeping internal state and observations in torch.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scene_factory: Callable[[], Scene],
        config: TorchORCAEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.scene_factory = scene_factory
        self.config = config if config is not None else TorchORCAEnvConfig()
        self.device = torch.device(str(self.config.device))

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
        max_goal_distance_termination = self.config.reward.max_goal_distance_termination
        if max_goal_distance_termination is not None and float(max_goal_distance_termination) <= 0.0:
            raise ValueError("reward.max_goal_distance_termination must be > 0 when set")

        occ_cfg = self.config.occupancy
        self._occupancy_resolution = float(occ_cfg.resolution)
        self._occupancy_resolution_xy = (self._occupancy_resolution, self._occupancy_resolution)
        self._occupancy_resolution_tensor = torch.tensor(
            [self._occupancy_resolution, self._occupancy_resolution],
            dtype=torch.float32,
            device=self.device,
        )
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
        self._goals: torch.Tensor | None = None
        self._last_positions: torch.Tensor | None = None
        self._last_velocities: torch.Tensor | None = None
        self._last_commanded_velocity: torch.Tensor | None = None
        self._scene_static_map: torch.Tensor | None = None
        self._scene_origin: tuple[float, float] | None = None
        self._scene_center: torch.Tensor | None = None
        self._scene_canvas_shape: tuple[int, int] | None = None
        self._dynamic_renderer: Occupancy2d | None = None
        self._dynamic_context_history: deque[torch.Tensor] = deque(maxlen=self._dynamic_context_len + 1)
        self._step_count: int = 0
        self._termination_checks: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
            ("success", self._terminate_on_success),
            ("too_far", self._terminate_on_too_far),
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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

        positions = torch.as_tensor(self.sim.get_agent_positions(), dtype=torch.float32, device=self.device)
        velocities = torch.as_tensor(self.sim.get_agent_velocities(), dtype=torch.float32, device=self.device)
        self._validate_controlled_agent_index(num_agents=int(positions.shape[0]))

        self._goals = torch.as_tensor([agent.goal for agent in self.sim.scene.agents], dtype=torch.float32, device=self.device)
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

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]:
        if self.sim is None or self._last_positions is None or self._last_velocities is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(-1)
        if action_t.numel() != 2:
            raise ValueError(f"Expected action with 2 elements, got shape {tuple(action_t.shape)}")

        low = torch.as_tensor(self.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(self.action_space.high, dtype=torch.float32, device=self.device)
        action_t = torch.clamp(action_t, min=low, max=high)

        commanded_velocity = action_t * self._controlled_agent_max_speed
        controlled_idx = int(self.config.controlled_agent_index)

        new_positions_np, new_velocities_np = self.sim.step(
            controlled_pref_velocities={controlled_idx: commanded_velocity.detach().cpu().numpy()},
            return_velocities=True,
        )
        new_positions = torch.as_tensor(new_positions_np, dtype=torch.float32, device=self.device)
        new_velocities = torch.as_tensor(new_velocities_np, dtype=torch.float32, device=self.device)

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
        self._last_commanded_velocity = commanded_velocity.clone()
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

    def _goal_distance(self, positions: torch.Tensor) -> float:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")
        controlled_idx = int(self.config.controlled_agent_index)
        diff = self._goals[controlled_idx] - positions[controlled_idx]
        return float(torch.linalg.norm(diff).item())

    def _initialize_occupancy(self, *, scene: Scene, positions: torch.Tensor) -> None:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")

        traj_seed = torch.stack([positions, self._goals], dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        scene_origin, scene_size, scene_canvas_shape, scene_center_np = _compute_scene_canvas(
            traj=traj_seed,
            obstacles=scene.obstacles,
            resolution=self._occupancy_resolution,
            occupancy_length=float(self.config.occupancy.patch_length),
            occupancy_width=float(self.config.occupancy.patch_width),
        )
        self._scene_origin = (float(scene_origin[0]), float(scene_origin[1]))
        self._scene_center = torch.as_tensor(scene_center_np, dtype=torch.float32, device=self.device)
        self._scene_canvas_shape = (int(scene_canvas_shape[0]), int(scene_canvas_shape[1]))

        scene_size_tensor = torch.tensor([float(scene_size[0]), float(scene_size[1])], dtype=torch.float32, device=self.device)
        static_renderer = Occupancy2d(
            resolution=self._occupancy_resolution_tensor,
            size=scene_size_tensor,
            trajectory=None,
            static_obstacles=scene.obstacles,
            agent_radius=float(self.config.occupancy.agent_radius),
        )
        # Generated directly on self.device by passing device-resident resolution/size/center.
        self._scene_static_map = static_renderer.generate(center_offset=self._scene_center)[0].to(
            device=self.device,
            dtype=torch.float32,
        )

        self._dynamic_renderer = Occupancy2d(
            resolution=self._occupancy_resolution_tensor,
            size=scene_size_tensor,
            trajectory=None,
            static_obstacles=None,
            agent_radius=float(self.config.occupancy.agent_radius),
        )
        self._dynamic_context_history.clear()

    def _build_obs(self, positions: torch.Tensor, velocities: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")
        if self._scene_static_map is None or self._scene_origin is None:
            raise RuntimeError("Occupancy state is not initialized")
        if self._scene_center is None or self._scene_canvas_shape is None or self._dynamic_renderer is None:
            raise RuntimeError("Dynamic occupancy state is not initialized")

        controlled_idx = int(self.config.controlled_agent_index)
        goal_offset = (self._goals[controlled_idx] - positions[controlled_idx]).to(dtype=torch.float32)
        current_velocity = velocities[controlled_idx].to(dtype=torch.float32)
        if self._last_commanded_velocity is None:
            last_commanded_velocity = torch.zeros(2, dtype=torch.float32, device=self.device)
        else:
            last_commanded_velocity = self._last_commanded_velocity.to(dtype=torch.float32)

        center_xy = positions[controlled_idx].to(dtype=torch.float32)
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
            other_positions = positions[other_indices].to(dtype=torch.float32).unsqueeze(0)
            self._dynamic_renderer.update_inputs(trajectory=other_positions, static_obstacles=[])
            dynamic_global = self._dynamic_renderer.generate(center_offset=self._scene_center)[0].to(
                device=self.device,
                dtype=torch.float32,
            )
        else:
            dynamic_global = torch.zeros(self._scene_canvas_shape, dtype=torch.float32, device=self.device)

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

        return {
            "dynamic_context": dynamic_context.to(dtype=torch.float32),
            "static_map": static_local.unsqueeze(0).to(dtype=torch.float32),
            "goal_position": goal_offset,
            "current_velocity": current_velocity,
            "last_commanded_velocity": last_commanded_velocity,
        }

    def _compute_reward_terminated(
        self,
        *,
        prev_positions: torch.Tensor,
        new_positions: torch.Tensor,
        new_velocities: torch.Tensor,
    ) -> tuple[float, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        _, new_goal_distance, progress = self._compute_progress(prev_positions, new_positions)
        collision = self._compute_collision(new_positions)
        controlled_speed, within_goal, stationary, success = self._compute_success_state(
            new_goal_distance=new_goal_distance,
            new_velocities=new_velocities,
        )
        too_far = self._compute_too_far(new_goal_distance)

        reward_terms = {
            "progress": self._compute_progress_reward(progress), #self._compute_distance_reward(new_goal_distance),
            "step_penalty": self._compute_step_penalty_reward(),
            "collision": self._compute_collision_reward(collision),
            "success": self._compute_success_reward(success, new_goal_distance),
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
            "too_far": bool(too_far),
            "success_speed_threshold": float(self.config.reward.success_speed_threshold),
            "max_goal_distance_termination": (
                None
                if self.config.reward.max_goal_distance_termination is None
                else float(self.config.reward.max_goal_distance_termination)
            ),
            "distance_to_goal": float(new_goal_distance),
            "reward_terms": reward_terms,
        }

        termination_state = {
            "success": bool(success),
            "too_far": bool(too_far),
            "collision": bool(collision),
            "goal_distance": float(new_goal_distance),
            "step_count": int(self._step_count),
        }
        terminated, termination_reasons = self._evaluate_termination_checks(termination_state)
        info["termination_reasons"] = termination_reasons
        return float(reward), bool(terminated), info

    def _compute_progress(self, prev_positions: torch.Tensor, new_positions: torch.Tensor) -> tuple[float, float, float]:
        prev_goal_distance = self._goal_distance(prev_positions)
        new_goal_distance = self._goal_distance(new_positions)
        progress = float(prev_goal_distance - new_goal_distance)
        return float(prev_goal_distance), float(new_goal_distance), progress

    def _compute_collision(self, new_positions: torch.Tensor) -> bool:
        controlled_idx = int(self.config.controlled_agent_index)
        controlled_pos = new_positions[controlled_idx]
        diffs = new_positions - controlled_pos.unsqueeze(0)
        dists = torch.linalg.norm(diffs, dim=1)
        self_mask = torch.zeros_like(dists, dtype=torch.bool)
        self_mask[controlled_idx] = True
        collision_distance = float(self.config.reward.collision_distance)
        return bool(torch.any((dists < collision_distance) & (~self_mask)).item())

    def _compute_success_state(
        self,
        *,
        new_goal_distance: float,
        new_velocities: torch.Tensor,
    ) -> tuple[float, bool, bool, bool]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        controlled_idx = int(self.config.controlled_agent_index)
        controlled_speed = float(torch.linalg.norm(new_velocities[controlled_idx]).item())
        within_goal = bool(new_goal_distance <= float(self.config.reward.success_distance))
        stationary = bool(controlled_speed <= float(self.config.reward.success_speed_threshold))
        success = bool(within_goal and stationary)
        return float(controlled_speed), bool(within_goal), bool(stationary), bool(success)

    def _compute_progress_reward(self, progress: float) -> float:
        penalty = min(progress, 0.0) * 2 * float(self.config.reward.progress_weight)
        reward = max(progress, 0.0) * float(self.config.reward.progress_weight)
        return penalty + reward
    
    def _compute_distance_reward(self, distance_to_goal: float) -> float:
        sigma = 5.0
        return float(self.config.reward.progress_weight) * (1 - math.tanh(distance_to_goal / sigma))

    def _compute_step_penalty_reward(self) -> float:
        return float(self.config.reward.step_penalty)

    def _compute_collision_reward(self, collision: bool) -> float:
        return float(self.config.reward.collision_penalty) if collision else 0.0

    def _compute_success_reward(self, success: bool, distance_to_goal: float) -> float:
        goal_bonus = 1.0 - math.tanh(distance_to_goal/3.0)
        return float(self.config.reward.success_reward) * goal_bonus * success

    def _compute_too_far(self, new_goal_distance: float) -> bool:
        threshold = self.config.reward.max_goal_distance_termination
        if threshold is None:
            return False
        return bool(float(new_goal_distance) >= float(threshold))

    def _evaluate_termination_checks(self, state: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for name, check in self._termination_checks:
            if bool(check(state)):
                reasons.append(name)
        return bool(reasons), reasons

    def _terminate_on_success(self, state: dict[str, Any]) -> bool:
        return bool(state.get("success", False))

    def _terminate_on_too_far(self, state: dict[str, Any]) -> bool:
        return bool(state.get("too_far", False))

    def _compute_action_change_penalty(self, commanded_velocity: torch.Tensor) -> tuple[float, float]:
        action_change = 0.0
        if self._last_commanded_velocity is not None:
            action_change = float(torch.linalg.norm(commanded_velocity - self._last_commanded_velocity).item())
        penalty = -float(self.config.reward.action_change_penalty_weight) * float(action_change)
        return float(action_change), float(penalty)


__all__ = [
    "TorchORCAEnv",
    "TorchORCAEnvConfig",
    "TorchORCAOccupancyConfig",
    "TorchORCARewardConfig",
    "TorchORCASimConfig",
]
