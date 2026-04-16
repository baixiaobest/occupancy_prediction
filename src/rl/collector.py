from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.scene import Scene

from .counterfactual import sample_random_velocity_plans
from .observation_manager import ObservationBatchContext, ObservationManager, build_observation_manager
from .replay_buffer import ReplayBuffer, TensorDict


@dataclass
class RandomPlanCollectorConfig:
    horizon: int
    num_candidates: int
    max_speed: float
    delta_std: float = 0.25
    dt: float = 0.1
    include_current_velocity_candidate: bool = True
    action_selection: Literal["uniform", "first"] = "uniform"
    seed: int = 0


@dataclass
class CollectSummary:
    transitions_added: int
    episodes_completed: int
    total_reward: float


class RandomPlanCollector:
    """Collect single-env rollouts with random candidate velocity plans."""

    def __init__(
        self,
        *,
        env: object,
        replay_buffer: ReplayBuffer,
        observation_manager: ObservationManager | None,
        config: RandomPlanCollectorConfig,
    ) -> None:
        if int(config.horizon) <= 0:
            raise ValueError("horizon must be > 0")
        if int(config.num_candidates) <= 0:
            raise ValueError("num_candidates must be > 0")
        if float(config.max_speed) <= 0.0:
            raise ValueError("max_speed must be > 0")
        if float(config.delta_std) < 0.0:
            raise ValueError("delta_std must be >= 0")
        if float(config.dt) <= 0.0:
            raise ValueError("dt must be > 0")
        if config.action_selection not in {"uniform", "first"}:
            raise ValueError("action_selection must be 'uniform' or 'first'")

        self.env = env
        self.replay_buffer = replay_buffer
        self.observation_manager = self._resolve_observation_manager(observation_manager)
        self.config = config
        self._prepared_obs: TensorDict | None = None
        self._seed = int(config.seed)
        self._rng: torch.Generator | None = None
        self._rng_device: torch.device | None = None

    def reset_episode(self, seed: int | None = None) -> TensorDict:
        self.observation_manager.reset()
        raw_obs = self.env.reset(seed=seed)
        scene = self._get_scene()
        self._prepared_obs = self.observation_manager.compute(
            ObservationBatchContext(raw_obs=raw_obs, scene=scene)
        )
        return self._prepared_obs

    def collect_steps(self, num_steps: int, *, reset_seed: int | None = None) -> CollectSummary:
        steps = int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be > 0")

        if self._prepared_obs is None:
            self.reset_episode(seed=reset_seed)

        transitions_added = 0
        episodes_completed = 0
        total_reward = 0.0

        for _ in range(steps):
            if self._prepared_obs is None:
                raise RuntimeError("Collector observation state is not initialized")

            obs = self._prepared_obs
            rng = self._get_rng(torch.as_tensor(obs["current_velocity"]).device)
            candidate_plans = sample_random_velocity_plans(
                current_velocity=obs["current_velocity"],
                num_candidates=int(self.config.num_candidates),
                horizon=int(self.config.horizon),
                max_speed=float(self.config.max_speed),
                delta_std=float(self.config.delta_std),
                dt=float(self.config.dt),
                include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
                generator=rng,
            )
            selected_plan, _selected_indices = self._select_plans(candidate_plans)
            next_raw_obs, rewards, dones, _infos = self.env.step(selected_plan[0, 0])
            next_obs = self.observation_manager.compute(
                ObservationBatchContext(raw_obs=next_raw_obs, scene=self._get_scene())
            )

            self.replay_buffer.add_batch(
                obs=obs,
                actions=selected_plan,
                rewards=torch.as_tensor(rewards, dtype=torch.float32),
                next_obs=next_obs,
                dones=torch.as_tensor(dones),
                candidate_actions=candidate_plans,
            )

            transitions_added += 1
            total_reward += float(torch.as_tensor(rewards, dtype=torch.float32).sum().item())

            if bool(torch.as_tensor(dones).reshape(-1)[0].item()):
                episodes_completed += 1
                self.reset_episode()
            else:
                self._prepared_obs = next_obs

        return CollectSummary(
            transitions_added=transitions_added,
            episodes_completed=episodes_completed,
            total_reward=total_reward,
        )

    def _select_plans(self, candidate_plans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_plans.ndim != 4 or candidate_plans.shape[-1] != 2:
            raise ValueError("candidate_plans must have shape (N_env, K, horizon, 2)")

        num_env = int(candidate_plans.shape[0])
        num_candidates = int(candidate_plans.shape[1])
        if self.config.action_selection == "first":
            indices = torch.zeros((num_env,), dtype=torch.int64, device=candidate_plans.device)
        else:
            rng = self._get_rng(candidate_plans.device)
            indices = torch.randint(
                low=0,
                high=num_candidates,
                size=(num_env,),
                generator=rng,
                device=candidate_plans.device,
            )

        gathered = candidate_plans[torch.arange(num_env, device=candidate_plans.device), indices]
        return gathered, indices

    def _get_rng(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self._seed)
            self._rng_device = device
        return self._rng

    def _get_scene(self) -> Scene:
        sim = getattr(self.env, "sim", None)
        scene = getattr(sim, "scene", None)
        if not isinstance(scene, Scene):
            raise RuntimeError("collector env must expose env.sim.scene as a Scene")
        return scene

    def _resolve_observation_manager(
        self,
        observation_manager: ObservationManager | None,
    ) -> ObservationManager:
        if observation_manager is not None:
            return observation_manager

        env_config = getattr(self.env, "env_config", None)
        observation_config = getattr(env_config, "observation", None)
        if observation_config is None:
            raise ValueError(
                "observation_manager must be provided when env.env_config.observation is not set"
            )
        return build_observation_manager(observation_config)


__all__ = [
    "CollectSummary",
    "RandomPlanCollector",
    "RandomPlanCollectorConfig",
]