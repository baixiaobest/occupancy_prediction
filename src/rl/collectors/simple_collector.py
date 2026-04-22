from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .collector_base import BaseActionCollector
from ..counterfactual import sample_random_velocity_plans
from ..managers.observation_manager import ObservationManager
from ..networks.simple_proposal_network import SimpleVelocityProposalNetwork
from ..networks.q_common import sample_action_indices_from_q_scores
from ..replay_buffer import ReplayBuffer, TensorDict


@dataclass
class SimpleQActionSelectionConfig:
    temperature: float = 1.0
    seed: int = 0


@dataclass
class SimpleRandomActionCollectorConfig:
    num_candidates: int
    max_speed: float
    delta_std: float = 0.25
    dt: float = 0.1
    include_current_velocity_candidate: bool = True
    action_selection: Literal["uniform", "first", "q_softmax"] = "uniform"
    seed: int = 0
    q_selection: SimpleQActionSelectionConfig | None = None


@dataclass
class SimpleActionCollectorConfig:
    num_candidates: int
    include_current_velocity_candidate: bool = True
    action_selection: Literal["uniform", "first", "q_softmax"] = "uniform"
    seed: int = 0
    q_selection: SimpleQActionSelectionConfig | None = None
    max_speed: float | None = None


@dataclass
class SimpleCollectSummary:
    transitions_added: int
    episodes_completed: int
    total_reward: float


@dataclass
class SimpleActionSelectionResult:
    selected_action: torch.Tensor
    selected_indices: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_log_probs: torch.Tensor | None


class _BaseSimpleActionCollector(BaseActionCollector):
    def __init__(
        self,
        *,
        env: object,
        replay_buffer: ReplayBuffer,
        observation_manager: ObservationManager | None,
        action_selection: Literal["uniform", "first", "q_softmax"],
        seed: int,
        q_selection: SimpleQActionSelectionConfig | None,
        q_network: nn.Module | None,
    ) -> None:
        if action_selection not in {"uniform", "first", "q_softmax"}:
            raise ValueError("action_selection must be 'uniform', 'first' or 'q_softmax'")
        if action_selection == "q_softmax" and (q_network is None or q_selection is None):
            raise ValueError("q_softmax selection requires q_network and q_selection config")
        if q_selection is not None and float(q_selection.temperature) <= 0.0:
            raise ValueError("q_selection.temperature must be > 0")

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            observation_manager=observation_manager,
            seed=int(seed),
            selection_seed=None if q_selection is None else int(q_selection.seed),
        )
        self.action_selection = action_selection
        self.q_selection = q_selection
        self.q_network = q_network

    def select_action(self, obs: TensorDict) -> SimpleActionSelectionResult:
        candidate_actions = self._sample_candidate_actions(obs)
        selected_action, selected_indices, candidate_log_probs = self._select_actions(obs, candidate_actions)
        return SimpleActionSelectionResult(
            selected_action=selected_action,
            selected_indices=selected_indices,
            candidate_actions=candidate_actions,
            candidate_log_probs=candidate_log_probs,
        )

    def _select_actions(
        self,
        obs: TensorDict,
        candidate_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if candidate_actions.ndim != 3 or candidate_actions.shape[-1] != 2:
            raise ValueError("candidate_actions must have shape (N_env, K, 2)")
        num_env, num_candidates = candidate_actions.shape[:2]
        if self.action_selection == "first":
            indices = torch.zeros((num_env,), dtype=torch.int64, device=candidate_actions.device)
            candidate_log_probs = None
        elif self.action_selection == "q_softmax":
            indices, candidate_log_probs = self._select_actions_with_q(obs, candidate_actions)
        else:
            indices = torch.randint(
                0,
                num_candidates,
                (num_env,),
                generator=self._get_rng(candidate_actions.device),
                device=candidate_actions.device,
            )
            candidate_log_probs = None

        selected = candidate_actions[torch.arange(num_env, device=candidate_actions.device), indices]
        return selected, indices, candidate_log_probs

    def _select_actions_with_q(
        self,
        obs: TensorDict,
        candidate_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_network is None or self.q_selection is None:
            raise RuntimeError("Q selection requires q_network and q_selection config")
        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
                batch_size, num_candidates = candidate_actions.shape[:2]
                current_velocity = obs["current_velocity"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
                goal_position = obs["goal_position"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
                flat_actions = candidate_actions.reshape(batch_size * num_candidates, -1)
                q_scores = self.q_network(
                    current_velocity=current_velocity,
                    goal_position=goal_position,
                    action=flat_actions,
                ).reshape(batch_size, num_candidates)
        finally:
            if was_training:
                self.q_network.train()

        selection_indices, selection_probs = sample_action_indices_from_q_scores(
            q_scores,
            temperature=float(self.q_selection.temperature),
            generator=self._get_selection_rng(q_scores.device),
        )
        return selection_indices, torch.log(selection_probs.clamp_min(1e-8))

    def _step_env(self, action_selection: SimpleActionSelectionResult):
        return self.env.step(action_selection.selected_action[0])

    def _actions_for_replay(self, action_selection: SimpleActionSelectionResult) -> torch.Tensor:
        return action_selection.selected_action

    def _candidate_actions_for_replay(self, action_selection: SimpleActionSelectionResult) -> torch.Tensor:
        return action_selection.candidate_actions

    def _build_collect_summary(
        self,
        *,
        transitions_added: int,
        episodes_completed: int,
        total_reward: float,
    ) -> SimpleCollectSummary:
        return SimpleCollectSummary(
            transitions_added=transitions_added,
            episodes_completed=episodes_completed,
            total_reward=total_reward,
        )

    def _sample_candidate_actions(self, obs: TensorDict) -> torch.Tensor:
        raise NotImplementedError


class SimpleRandomActionCollector(_BaseSimpleActionCollector):
    def __init__(
        self,
        *,
        env: object,
        replay_buffer: ReplayBuffer,
        observation_manager: ObservationManager | None,
        config: SimpleRandomActionCollectorConfig,
        q_network: nn.Module | None = None,
    ) -> None:
        if int(config.num_candidates) <= 0:
            raise ValueError("num_candidates must be > 0")
        if float(config.max_speed) <= 0.0:
            raise ValueError("max_speed must be > 0")
        if float(config.delta_std) < 0.0:
            raise ValueError("delta_std must be >= 0")
        if float(config.dt) <= 0.0:
            raise ValueError("dt must be > 0")

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            observation_manager=observation_manager,
            action_selection=config.action_selection,
            seed=int(config.seed),
            q_selection=config.q_selection,
            q_network=q_network,
        )
        self.config = config

    def _sample_candidate_actions(self, obs: TensorDict) -> torch.Tensor:
        rng = self._get_rng(torch.as_tensor(obs["current_velocity"]).device)
        plans = sample_random_velocity_plans(
            current_velocity=obs["current_velocity"],
            num_candidates=int(self.config.num_candidates),
            horizon=1,
            max_speed=float(self.config.max_speed),
            delta_std=float(self.config.delta_std),
            dt=float(self.config.dt),
            include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
            generator=rng,
        )
        return plans[:, :, 0, :]

class SimpleActionCollector(_BaseSimpleActionCollector):
    def __init__(
        self,
        *,
        env: object,
        replay_buffer: ReplayBuffer,
        observation_manager: ObservationManager | None,
        config: SimpleActionCollectorConfig,
        proposal_network: SimpleVelocityProposalNetwork,
        q_network: nn.Module | None = None,
    ) -> None:
        if int(config.num_candidates) <= 0:
            raise ValueError("num_candidates must be > 0")
        if config.max_speed is not None and float(config.max_speed) <= 0.0:
            raise ValueError("max_speed must be > 0 when provided")

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            observation_manager=observation_manager,
            action_selection=config.action_selection,
            seed=int(config.seed),
            q_selection=config.q_selection,
            q_network=q_network,
        )
        self.config = config
        self.proposal_network = proposal_network

    def _sample_candidate_actions(self, obs: TensorDict) -> torch.Tensor:
        current_velocity = torch.as_tensor(obs["current_velocity"])
        was_training = self.proposal_network.training
        self.proposal_network.eval()
        try:
            with torch.no_grad():
                return self.proposal_network.sample_actions(
                    current_velocity=current_velocity,
                    goal_position=obs["goal_position"],
                    num_candidates=int(self.config.num_candidates),
                    include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
                    generator=self._get_rng(current_velocity.device),
                    max_speed=self.config.max_speed,
                )
        finally:
            if was_training:
                self.proposal_network.train()

    def _select_actions(
        self,
        obs: TensorDict,
        candidate_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return super()._select_actions(obs, candidate_actions)


__all__ = [
    "SimpleActionCollector",
    "SimpleActionCollectorConfig",
    "SimpleActionSelectionResult",
    "SimpleCollectSummary",
    "SimpleQActionSelectionConfig",
    "SimpleRandomActionCollector",
    "SimpleRandomActionCollectorConfig",
]