from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.ORCASim import ORCASim
from src.scene import Scene
from src.rl.managers.observation_manager import ObservationConfig
from src.rl.managers.reward_manager import (
    RewardBatchContext,
    RewardConfig,
    RewardManager,
    RewardTermCfg,
    build_reward_manager,
)


@dataclass
class ORCASimConfig:
    """Configuration forwarded to ORCASim constructor."""

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
class SingleEnvConfig:
    """Single-environment wrapper settings."""

    max_steps: int = 200
    controlled_agent_index: int = 0
    device: str = "cpu"
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig | None = None


def _build_default_reward_manager(reward_cfg: RewardConfig) -> RewardManager:
    """Build default reward manager equivalent to the legacy hardcoded reward."""
    return build_reward_manager(reward_cfg)


class ORCASingleEnv:
    """Single RL environment wrapper around ORCASim.

    API is intentionally vector-ready:
    - observations contain leading env axis with N_env=1
    - rewards and dones have shape (1,)
    """

    def __init__(
        self,
        *,
        scene_factory: Callable[[], Scene],
        sim_config: ORCASimConfig,
        env_config: SingleEnvConfig,
        reward_manager: RewardManager | None = None,
    ) -> None:
        self.scene_factory = scene_factory
        self.sim_config = sim_config
        self.env_config = env_config

        self.sim: ORCASim | None = None
        self.reward_manager = reward_manager
        self.device = torch.device(self.env_config.device)
        self._step_count = 0
        self._last_positions: torch.Tensor | None = None
        self._last_velocities: torch.Tensor | None = None
        self._goals: torch.Tensor | None = None

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Start a fresh scene and return initial observation."""
        scene = self.scene_factory()
        kwargs: dict[str, Any] = {
            "time_step": self.sim_config.time_step,
            "neighbor_dist": self.sim_config.neighbor_dist,
            "max_neighbors": self.sim_config.max_neighbors,
            "time_horizon": self.sim_config.time_horizon,
            "time_horizon_obst": self.sim_config.time_horizon_obst,
            "radius": self.sim_config.radius,
            "max_speed": self.sim_config.max_speed,
            "goal_tolerance": self.sim_config.goal_tolerance,
            "path_goal_switch_tolerance": self.sim_config.path_goal_switch_tolerance,
            "path_segment_remaining_switch_ratio": self.sim_config.path_segment_remaining_switch_ratio,
            "pref_velocity_noise_std": self.sim_config.pref_velocity_noise_std,
            "pref_velocity_noise_interval": self.sim_config.pref_velocity_noise_interval,
            "pref_velocity_noise_seed": self.sim_config.pref_velocity_noise_seed,
            "lateral_control_gain": self.sim_config.lateral_control_gain,
            "lateral_control_max_speed": self.sim_config.lateral_control_max_speed,
        }
        if self.sim_config.strict_controlled_agent:
            kwargs["strict_controlled_agent_index"] = int(self.env_config.controlled_agent_index)
            kwargs["strict_control_velocity_tolerance"] = float(self.sim_config.strict_control_velocity_tolerance)
            kwargs["strict_control_assert"] = bool(self.sim_config.strict_control_assert)
        if seed is not None:
            kwargs["region_pair_seed"] = int(seed)

        self.sim = ORCASim(scene=scene, **kwargs)
        self._step_count = 0

        self._last_positions = torch.from_numpy(self.sim.get_agent_positions()).to(device=self.device)
        self._last_velocities = torch.from_numpy(self.sim.get_agent_velocities()).to(device=self.device)
        self._validate_controlled_index(num_agents=int(self._last_positions.shape[0]))
        self._goals = torch.as_tensor(
            [agent.goal for agent in self.sim.scene.agents],
            dtype=torch.float32,
            device=self.device,
        )

        if self.reward_manager is None:
            self.reward_manager = _build_default_reward_manager(self.env_config.reward)

        return self._build_obs(self._last_positions, self._last_velocities)

    def step(
        self,
        action_velocity: torch.Tensor | Any,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Apply one controlled-agent action and advance simulation by one step."""
        if self.sim is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._last_positions is None or self._last_velocities is None:
            raise RuntimeError("Missing previous simulator state; call reset() first.")

        controlled_idx = int(self.env_config.controlled_agent_index)
        action = torch.as_tensor(action_velocity, dtype=torch.float32, device=self.device)
        if tuple(action.shape) != (2,):
            raise ValueError("action_velocity must have shape (2,)")

        positions_np, velocities_np = self.sim.step(
            controlled_pref_velocities={controlled_idx: action.detach().cpu().numpy()},
            return_velocities=True,
        )
        positions = torch.from_numpy(positions_np).to(device=self.device)
        velocities = torch.from_numpy(velocities_np).to(device=self.device)

        rewards_vec, done_vec, info = self._compute_reward_done_info(
            prev_positions=self._last_positions,
            new_positions=positions,
            controlled_idx=controlled_idx,
        )

        self._last_positions = positions
        self._last_velocities = velocities
        self._step_count += 1

        if self._step_count >= int(self.env_config.max_steps):
            done_vec[:] = True
            info["timeout"] = True

        obs = self._build_obs(positions, velocities)
        rewards = rewards_vec.to(dtype=torch.float32)
        dones = done_vec.to(dtype=torch.bool)
        infos = [info]
        return obs, rewards, dones, infos

    def _validate_controlled_index(self, *, num_agents: int) -> None:
        idx = int(self.env_config.controlled_agent_index)
        if idx < 0 or idx >= num_agents:
            raise ValueError(
                f"controlled_agent_index {idx} is out of range for {num_agents} agents"
            )

    def _build_obs(self, positions: torch.Tensor, velocities: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")

        controlled_idx = int(self.env_config.controlled_agent_index)

        # Keep an explicit env axis so collector/replay code remains vector-ready.
        return {
            "positions": positions.unsqueeze(0),
            "velocities": velocities.unsqueeze(0),
            "goals": self._goals.unsqueeze(0),
            "controlled_agent_index": torch.tensor([controlled_idx], dtype=torch.int64, device=self.device),
            "step_count": torch.tensor([self._step_count], dtype=torch.int64, device=self.device),
        }

    def _compute_reward_done_info(
        self,
        *,
        prev_positions: torch.Tensor,
        new_positions: torch.Tensor,
        controlled_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        if self.reward_manager is None:
            raise RuntimeError("Reward manager is not initialized")
        if self._goals is None:
            raise RuntimeError("Goals are not initialized")

        context = RewardBatchContext(
            prev_positions=prev_positions.unsqueeze(0),
            new_positions=new_positions.unsqueeze(0),
            goals=self._goals.unsqueeze(0),
            controlled_agent_indices=torch.tensor([controlled_idx], dtype=torch.int64, device=self.device),
            goal_tolerances=torch.tensor([self.sim.goal_tolerance], dtype=torch.float32, device=self.device),
        )

        total_reward, weighted_terms, raw_terms = self.reward_manager.compute(context)

        goal_distance = context.controlled_new_goal_distance()
        success = goal_distance <= context.goal_tolerances

        collision_term = raw_terms.get("collision")
        collision = bool(collision_term[0].item() > 0.5) if collision_term is not None else False

        progress_term = raw_terms.get("progress")
        progress = float(progress_term[0].item()) if progress_term is not None else 0.0

        done = success.to(dtype=torch.bool)
        info: dict[str, Any] = {
            "success": bool(success[0].item()),
            "collision": bool(collision),
            "timeout": False,
            "goal_distance": float(goal_distance[0].item()),
            "progress": float(progress),
            "reward_terms": {name: float(values[0].item()) for name, values in weighted_terms.items()},
        }
        return total_reward.to(dtype=torch.float32), done, info


__all__ = [
    "ORCASimConfig",
    "RewardConfig",
    "RewardTermCfg",
    "SingleEnvConfig",
    "ORCASingleEnv",
]
