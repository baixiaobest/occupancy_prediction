from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.scene import Scene

from .counterfactual import CounterfactualRolloutBatch, rollout_counterfactual_futures, sample_random_velocity_plans
from .observation_manager import ObservationBatchContext, ObservationManager, build_observation_manager
from .q_trainer import sample_action_indices_from_q_scores
from .replay_buffer import ReplayBuffer, TensorDict


@dataclass
class QActionSelectionConfig:
    temperature: float = 1.0
    seed: int = 0
    tap_layer: int = 1
    latent_channels: int = 4
    latent_shape: tuple[int, int, int] = (1, 2, 2)
    binary_feedback: bool = False
    threshold: float = 0.5


@dataclass
class RandomPlanCollectorConfig:
    horizon: int
    num_candidates: int
    max_speed: float
    delta_std: float = 0.25
    dt: float = 0.1
    include_current_velocity_candidate: bool = True
    action_selection: Literal["uniform", "first", "q_softmax"] = "uniform"
    seed: int = 0
    q_selection: QActionSelectionConfig | None = None


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
        q_network: nn.Module | None = None,
        decoder: nn.Module | None = None,
        candidate_sampler=sample_random_velocity_plans,
        counterfactual_rollout_fn=rollout_counterfactual_futures,
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
        if config.action_selection not in {"uniform", "first", "q_softmax"}:
            raise ValueError("action_selection must be 'uniform', 'first' or 'q_softmax'")
        if config.action_selection == "q_softmax":
            if q_network is None:
                raise ValueError("q_network must be provided when action_selection='q_softmax'")
            if decoder is None:
                raise ValueError("decoder must be provided when action_selection='q_softmax'")
            if config.q_selection is None:
                raise ValueError("config.q_selection must be provided when action_selection='q_softmax'")
            if float(config.q_selection.temperature) <= 0.0:
                raise ValueError("q_selection.temperature must be > 0")
            if int(config.q_selection.tap_layer) <= 0:
                raise ValueError("q_selection.tap_layer must be > 0")
            if int(config.q_selection.latent_channels) <= 0:
                raise ValueError("q_selection.latent_channels must be > 0")
            if len(config.q_selection.latent_shape) != 3 or any(int(v) <= 0 for v in config.q_selection.latent_shape):
                raise ValueError("q_selection.latent_shape must contain three positive ints")
            if not (0.0 <= float(config.q_selection.threshold) <= 1.0):
                raise ValueError("q_selection.threshold must be in [0, 1]")

        self.env = env
        self.replay_buffer = replay_buffer
        self.observation_manager = self._resolve_observation_manager(observation_manager)
        self.config = config
        self.q_network = q_network
        self.decoder = decoder
        self.candidate_sampler = candidate_sampler
        self.counterfactual_rollout_fn = counterfactual_rollout_fn
        self._prepared_obs: TensorDict | None = None
        self._seed = int(config.seed)
        self._rng: torch.Generator | None = None
        self._rng_device: torch.device | None = None
        self._selection_rng: torch.Generator | None = None
        self._selection_rng_device: torch.device | None = None

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
            candidate_plans = self.candidate_sampler(
                current_velocity=obs["current_velocity"],
                num_candidates=int(self.config.num_candidates),
                horizon=int(self.config.horizon),
                max_speed=float(self.config.max_speed),
                delta_std=float(self.config.delta_std),
                dt=float(self.config.dt),
                include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
                generator=rng,
            )
            selected_plan, _selected_indices, candidate_log_probs = self._select_plans(obs, candidate_plans)
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
                candidate_log_probs=candidate_log_probs,
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

    def _select_plans(
        self,
        obs: TensorDict,
        candidate_plans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if candidate_plans.ndim != 4 or candidate_plans.shape[-1] != 2:
            raise ValueError("candidate_plans must have shape (N_env, K, horizon, 2)")

        num_env = int(candidate_plans.shape[0])
        num_candidates = int(candidate_plans.shape[1])
        if self.config.action_selection == "first":
            indices = torch.zeros((num_env,), dtype=torch.int64, device=candidate_plans.device)
            candidate_log_probs = None
        elif self.config.action_selection == "q_softmax":
            indices, candidate_log_probs = self._select_plans_with_q(obs, candidate_plans)
        else:
            rng = self._get_rng(candidate_plans.device)
            indices = torch.randint(
                low=0,
                high=num_candidates,
                size=(num_env,),
                generator=rng,
                device=candidate_plans.device,
            )
            candidate_log_probs = None

        gathered = candidate_plans[torch.arange(num_env, device=candidate_plans.device), indices]
        return gathered, indices, candidate_log_probs

    def _select_plans_with_q(
        self,
        obs: TensorDict,
        candidate_plans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_network is None or self.decoder is None or self.config.q_selection is None:
            raise RuntimeError("Q selection requires q_network, decoder, and config.q_selection")

        rollout = self._rollout_candidates(obs, candidate_plans)
        planned_velocities, tapped_features = rollout.flatten_candidates_for_q()
        if tapped_features is None:
            raise RuntimeError("Counterfactual rollout must return tapped features for Q-based selection")

        batch_size, num_candidates = candidate_plans.shape[:2]
        goal_position = obs["goal_position"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        current_velocity = obs["current_velocity"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        q_scores = self.q_network(
            goal_position=goal_position,
            current_velocity=current_velocity,
            planned_velocities=planned_velocities,
            tapped_future_features=tapped_features,
        ).reshape(batch_size, num_candidates)

        selection_indices, selection_probs = sample_action_indices_from_q_scores(
            q_scores,
            temperature=float(self.config.q_selection.temperature),
            generator=self._get_selection_rng(q_scores.device),
        )
        return selection_indices, torch.log(selection_probs.clamp_min(1e-8))

    def _get_rng(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self._seed)
            self._rng_device = device
        return self._rng

    def _get_selection_rng(self, device: torch.device) -> torch.Generator:
        if self.config.q_selection is None:
            raise RuntimeError("selection RNG requested without q_selection config")
        if self._selection_rng is None or self._selection_rng_device != device:
            self._selection_rng = torch.Generator(device=device)
            self._selection_rng.manual_seed(int(self.config.q_selection.seed))
            self._selection_rng_device = device
        return self._selection_rng

    def _rollout_candidates(
        self,
        obs: TensorDict,
        candidate_velocity_plans: torch.Tensor,
    ) -> CounterfactualRolloutBatch:
        if self.decoder is None or self.config.q_selection is None:
            raise RuntimeError("Candidate rollout requested without decoder or q_selection config")
        return self.counterfactual_rollout_fn(
            decoder=self.decoder,
            dynamic_context=obs["dynamic_context"],
            static_map=obs["static_map"],
            candidate_velocity_plans=candidate_velocity_plans,
            latent_channels=int(self.config.q_selection.latent_channels),
            latent_shape=tuple(int(v) for v in self.config.q_selection.latent_shape),
            dt=float(self.config.dt),
            current_position_offset=obs.get("current_position_offset"),
            tap_layer=int(self.config.q_selection.tap_layer),
            binary_feedback=bool(self.config.q_selection.binary_feedback),
            threshold=float(self.config.q_selection.threshold),
        )

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
    "QActionSelectionConfig",
    "RandomPlanCollector",
    "RandomPlanCollectorConfig",
]