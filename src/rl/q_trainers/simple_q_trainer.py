from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from ..counterfactual import sample_random_velocity_plans
from .q_trainer_base import BaseRandomCandidateQTrainer, validate_common_random_candidate_q_config
from ..replay_buffer import ReplaySampleBatch


@dataclass
class SimpleQTrainerConfig:
    discount: float = 0.99
    target_tau: float = 0.01
    selection_temperature: float = 1.0
    selection_seed: int = 0
    num_bootstrap_candidates: int = 8
    max_speed: float = 2.0
    delta_std: float = 0.25
    dt: float = 0.1
    include_current_velocity_candidate: bool = True
    grad_clip_norm: float | None = None
    loss_type: Literal["mse", "smooth_l1"] = "smooth_l1"


@dataclass
class SimpleQTrainStepStats:
    loss: float
    q_pred_mean: float
    target_mean: float
    next_q_mean: float
    selection_entropy_mean: float
    reward_mean: float
    done_fraction: float


class SimpleRandomCandidateQTrainer(BaseRandomCandidateQTrainer):
    def __init__(
        self,
        *,
        q_network: nn.Module,
        target_q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SimpleQTrainerConfig,
    ) -> None:
        validate_common_random_candidate_q_config(config)
        super().__init__(
            q_network=q_network,
            target_q_network=target_q_network,
            optimizer=optimizer,
            config=config,
        )

    def _compute_q_pred(self, batch: ReplaySampleBatch) -> torch.Tensor:
        return self.q_network(
            current_velocity=batch.obs["current_velocity"],
            goal_position=batch.obs["goal_position"],
            action=batch.actions,
        )

    def _compute_td_target(self, batch: ReplaySampleBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_candidate_actions = sample_random_velocity_plans(
            current_velocity=batch.next_obs["current_velocity"],
            num_candidates=int(self.config.num_bootstrap_candidates),
            horizon=1,
            max_speed=float(self.config.max_speed),
            delta_std=float(self.config.delta_std),
            dt=float(self.config.dt),
            include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
        )[:, :, 0, :]
        batch_size, num_candidates = next_candidate_actions.shape[:2]
        current_velocity = batch.next_obs["current_velocity"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        goal_position = batch.next_obs["goal_position"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        flat_actions = next_candidate_actions.reshape(batch_size * num_candidates, -1)

        was_training = self.target_q_network.training
        self.target_q_network.eval()
        try:
            next_q_scores = self.target_q_network(
                current_velocity=current_velocity,
                goal_position=goal_position,
                action=flat_actions,
            ).reshape(batch_size, num_candidates)
        finally:
            if was_training:
                self.target_q_network.train()

        return self._build_td_target_from_next_q_scores(batch, next_q_scores)

    @staticmethod
    def _validate_batch(batch: ReplaySampleBatch) -> None:
        required_obs_keys = {"current_velocity", "goal_position"}
        missing_obs_keys = required_obs_keys.difference(batch.obs.keys())
        if missing_obs_keys:
            raise ValueError(f"Replay batch obs is missing required keys: {sorted(missing_obs_keys)}")
        missing_next_obs_keys = required_obs_keys.difference(batch.next_obs.keys())
        if missing_next_obs_keys:
            raise ValueError(f"Replay batch next_obs is missing required keys: {sorted(missing_next_obs_keys)}")
        if batch.actions.ndim != 2 or batch.actions.shape[1] != 2:
            raise ValueError("batch.actions must have shape (B, 2)")

    @staticmethod
    def _build_train_step_stats(
        *,
        loss: float,
        q_pred_mean: float,
        target_mean: float,
        next_q_mean: float,
        selection_entropy_mean: float,
        reward_mean: float,
        done_fraction: float,
    ) -> SimpleQTrainStepStats:
        return SimpleQTrainStepStats(
            loss=loss,
            q_pred_mean=q_pred_mean,
            target_mean=target_mean,
            next_q_mean=next_q_mean,
            selection_entropy_mean=selection_entropy_mean,
            reward_mean=reward_mean,
            done_fraction=done_fraction,
        )


__all__ = [
    "SimpleQTrainerConfig",
    "SimpleQTrainStepStats",
    "SimpleRandomCandidateQTrainer",
]