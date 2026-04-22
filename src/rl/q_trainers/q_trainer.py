from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from ..counterfactual import CounterfactualRolloutBatch, rollout_counterfactual_futures, sample_random_velocity_plans
from ..networks.q_common import q_scores_to_probabilities, sample_action_indices_from_q_scores, soft_update_module
from .q_trainer_base import BaseQTrainer, validate_common_random_candidate_q_config
from ..replay_buffer import ReplaySampleBatch


CandidateSamplerFn = Callable[..., torch.Tensor]
CounterfactualRolloutFn = Callable[..., CounterfactualRolloutBatch]


@dataclass
class QTrainerConfig:
    discount: float = 0.99
    target_tau: float = 0.01
    selection_temperature: float = 1.0
    selection_seed: int = 0
    num_bootstrap_candidates: int = 8
    max_speed: float = 2.0
    delta_std: float = 0.25
    dt: float = 0.1
    tap_layer: int = 1
    latent_channels: int = 4
    latent_shape: tuple[int, int, int] = (1, 2, 2)
    include_current_velocity_candidate: bool = True
    binary_feedback: bool = False
    threshold: float = 0.5
    grad_clip_norm: float | None = None
    loss_type: Literal["mse", "smooth_l1"] = "smooth_l1"


@dataclass
class QTrainStepStats:
    loss: float
    q_pred_mean: float
    target_mean: float
    next_q_mean: float
    selection_entropy_mean: float
    reward_mean: float
    done_fraction: float


class RandomCandidateQTrainer(BaseQTrainer):
    """Train a Q-network using random bootstrap candidates and decoder taps."""

    def __init__(
        self,
        *,
        q_network: nn.Module,
        target_q_network: nn.Module,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: QTrainerConfig,
        candidate_sampler: CandidateSamplerFn = sample_random_velocity_plans,
        counterfactual_rollout_fn: CounterfactualRolloutFn = rollout_counterfactual_futures,
    ) -> None:
        validate_common_random_candidate_q_config(config)
        if int(config.tap_layer) <= 0:
            raise ValueError("tap_layer must be > 0")
        if int(config.latent_channels) <= 0:
            raise ValueError("latent_channels must be > 0")
        if len(config.latent_shape) != 3 or any(int(v) <= 0 for v in config.latent_shape):
            raise ValueError("latent_shape must contain three positive ints")
        if not (0.0 <= float(config.threshold) <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        super().__init__(
            q_network=q_network,
            target_q_network=target_q_network,
            optimizer=optimizer,
            config=config,
        )
        self.decoder = decoder
        self.candidate_sampler = candidate_sampler
        self.counterfactual_rollout_fn = counterfactual_rollout_fn

    def _compute_q_pred(self, batch: ReplaySampleBatch) -> torch.Tensor:
        selected_taps = self._compute_selected_action_taps(batch)
        return self.q_network(
            goal_position=batch.obs["goal_position"],
            current_velocity=batch.obs["current_velocity"],
            planned_velocities=batch.actions,
            tapped_future_features=selected_taps,
        )

    def _compute_td_target(
        self,
        batch: ReplaySampleBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_obs = batch.next_obs
        next_candidate_plans = self.candidate_sampler(
            current_velocity=next_obs["current_velocity"],
            num_candidates=int(self.config.num_bootstrap_candidates),
            horizon=int(batch.actions.shape[1]),
            max_speed=float(self.config.max_speed),
            delta_std=float(self.config.delta_std),
            dt=float(self.config.dt),
            include_current_velocity_candidate=bool(self.config.include_current_velocity_candidate),
        )
        next_rollout = self._rollout_candidates(next_obs, next_candidate_plans)
        next_plans_flat, next_taps_flat = next_rollout.flatten_candidates_for_q()
        if next_taps_flat is None:
            raise RuntimeError("Counterfactual rollout must return tapped features for target computation")

        batch_size, num_candidates = next_rollout.candidate_velocity_plans.shape[:2]
        next_goal_flat = next_obs["goal_position"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        next_velocity_flat = next_obs["current_velocity"][:, None, :].expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        was_training = self.target_q_network.training
        self.target_q_network.eval()
        try:
            next_q_flat = self.target_q_network(
                goal_position=next_goal_flat,
                current_velocity=next_velocity_flat,
                planned_velocities=next_plans_flat,
                tapped_future_features=next_taps_flat,
            )
        finally:
            if was_training:
                self.target_q_network.train()
        next_q_scores = next_q_flat.reshape(batch_size, num_candidates)
        return self._build_td_target_from_next_q_scores(batch, next_q_scores)

    def _compute_selected_action_taps(self, batch: ReplaySampleBatch) -> torch.Tensor:
        selected_rollout = self._rollout_candidates(batch.obs, batch.actions.unsqueeze(1))
        if selected_rollout.tapped_features is None:
            raise RuntimeError("Counterfactual rollout must return tapped features for Q evaluation")
        return selected_rollout.tapped_features[:, 0]

    def _rollout_candidates(
        self,
        obs: dict[str, torch.Tensor],
        candidate_velocity_plans: torch.Tensor,
    ) -> CounterfactualRolloutBatch:
        return self.counterfactual_rollout_fn(
            decoder=self.decoder,
            dynamic_context=obs["dynamic_context"],
            static_map=obs["static_map"],
            candidate_velocity_plans=candidate_velocity_plans,
            latent_channels=int(self.config.latent_channels),
            latent_shape=tuple(int(v) for v in self.config.latent_shape),
            dt=float(self.config.dt),
            current_position_offset=obs.get("current_position_offset"),
            tap_layer=int(self.config.tap_layer),
            binary_feedback=bool(self.config.binary_feedback),
            threshold=float(self.config.threshold),
        )

    @staticmethod
    def _validate_batch(batch: ReplaySampleBatch) -> None:
        required_obs_keys = {"dynamic_context", "static_map", "current_velocity", "goal_position"}
        missing_obs_keys = required_obs_keys.difference(batch.obs.keys())
        if missing_obs_keys:
            raise ValueError(f"Replay batch obs is missing required keys: {sorted(missing_obs_keys)}")
        missing_next_obs_keys = required_obs_keys.difference(batch.next_obs.keys())
        if missing_next_obs_keys:
            raise ValueError(f"Replay batch next_obs is missing required keys: {sorted(missing_next_obs_keys)}")
        if batch.actions.ndim != 3 or batch.actions.shape[-1] != 2:
            raise ValueError("batch.actions must have shape (B, horizon, 2)")

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
    ) -> QTrainStepStats:
        return QTrainStepStats(
            loss=loss,
            q_pred_mean=q_pred_mean,
            target_mean=target_mean,
            next_q_mean=next_q_mean,
            selection_entropy_mean=selection_entropy_mean,
            reward_mean=reward_mean,
            done_fraction=done_fraction,
        )


__all__ = [
    "CandidateSamplerFn",
    "CounterfactualRolloutFn",
    "QTrainerConfig",
    "QTrainStepStats",
    "RandomCandidateQTrainer",
    "q_scores_to_probabilities",
    "sample_action_indices_from_q_scores",
    "soft_update_module",
]