from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .collector_base import BaseRandomActionCollector
from .counterfactual import CounterfactualRolloutBatch, rollout_counterfactual_futures, sample_random_velocity_plans
from .observation_manager import ObservationBatchContext, ObservationManager, build_observation_manager
from .q_common import sample_action_indices_from_q_scores
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


@dataclass
class ActionSelectionResult:
    selected_plan: torch.Tensor
    selected_indices: torch.Tensor
    candidate_plans: torch.Tensor
    candidate_log_probs: torch.Tensor | None


class RandomPlanCollector(BaseRandomActionCollector):
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

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            observation_manager=observation_manager,
            seed=int(config.seed),
            selection_seed=None if config.q_selection is None else int(config.q_selection.seed),
        )
        self.config = config
        self.q_network = q_network
        self.decoder = decoder
        self.candidate_sampler = candidate_sampler
        self.counterfactual_rollout_fn = counterfactual_rollout_fn

    def select_action(self, obs: TensorDict) -> ActionSelectionResult:
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
        selected_plan, selected_indices, candidate_log_probs = self._select_plans(obs, candidate_plans)
        return ActionSelectionResult(
            selected_plan=selected_plan,
            selected_indices=selected_indices,
            candidate_plans=candidate_plans,
            candidate_log_probs=candidate_log_probs,
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

        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
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
        finally:
            if was_training:
                self.q_network.train()

        selection_indices, selection_probs = sample_action_indices_from_q_scores(
            q_scores,
            temperature=float(self.config.q_selection.temperature),
            generator=self._get_selection_rng(q_scores.device),
        )
        return selection_indices, torch.log(selection_probs.clamp_min(1e-8))

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

    def _step_env(self, action_selection: ActionSelectionResult):
        return self.env.step(action_selection.selected_plan[0, 0])

    def _actions_for_replay(self, action_selection: ActionSelectionResult) -> torch.Tensor:
        return action_selection.selected_plan

    def _candidate_actions_for_replay(self, action_selection: ActionSelectionResult) -> torch.Tensor:
        return action_selection.candidate_plans

    def _build_collect_summary(
        self,
        *,
        transitions_added: int,
        episodes_completed: int,
        total_reward: float,
    ) -> CollectSummary:
        return CollectSummary(
            transitions_added=transitions_added,
            episodes_completed=episodes_completed,
            total_reward=total_reward,
        )


__all__ = [
    "ActionSelectionResult",
    "CollectSummary",
    "QActionSelectionConfig",
    "RandomPlanCollector",
    "RandomPlanCollectorConfig",
]