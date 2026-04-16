from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .counterfactual import CounterfactualRolloutBatch, rollout_counterfactual_futures, sample_random_velocity_plans
from .replay_buffer import ReplaySampleBatch


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


def soft_update_module(
    source: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    update_rate = float(tau)
    if not (0.0 <= update_rate <= 1.0):
        raise ValueError("tau must be in [0, 1]")

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - update_rate).add_(source_param, alpha=update_rate)


def q_scores_to_probabilities(
    scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    logits = torch.as_tensor(scores, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("scores must have shape (B, K)")
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(logits / temp, dim=1)


def sample_action_indices_from_q_scores(
    scores: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = q_scores_to_probabilities(scores, temperature=temperature)
    indices = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(1)
    return indices, probs


class RandomCandidateQTrainer:
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
        if not (0.0 <= float(config.discount) <= 1.0):
            raise ValueError("discount must be in [0, 1]")
        if not (0.0 <= float(config.target_tau) <= 1.0):
            raise ValueError("target_tau must be in [0, 1]")
        if float(config.selection_temperature) <= 0.0:
            raise ValueError("selection_temperature must be > 0")
        if int(config.num_bootstrap_candidates) <= 0:
            raise ValueError("num_bootstrap_candidates must be > 0")
        if float(config.max_speed) <= 0.0:
            raise ValueError("max_speed must be > 0")
        if float(config.delta_std) < 0.0:
            raise ValueError("delta_std must be >= 0")
        if float(config.dt) <= 0.0:
            raise ValueError("dt must be > 0")
        if int(config.tap_layer) <= 0:
            raise ValueError("tap_layer must be > 0")
        if int(config.latent_channels) <= 0:
            raise ValueError("latent_channels must be > 0")
        if len(config.latent_shape) != 3 or any(int(v) <= 0 for v in config.latent_shape):
            raise ValueError("latent_shape must contain three positive ints")
        if not (0.0 <= float(config.threshold) <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if config.grad_clip_norm is not None and float(config.grad_clip_norm) <= 0.0:
            raise ValueError("grad_clip_norm must be > 0 when provided")
        if config.loss_type not in {"mse", "smooth_l1"}:
            raise ValueError("loss_type must be 'mse' or 'smooth_l1'")

        self.q_network = q_network
        self.target_q_network = target_q_network
        self.decoder = decoder
        self.optimizer = optimizer
        self.config = config
        self.candidate_sampler = candidate_sampler
        self.counterfactual_rollout_fn = counterfactual_rollout_fn
        self._selection_rng: torch.Generator | None = None
        self._selection_rng_device: torch.device | None = None

    def train_step(self, batch: ReplaySampleBatch) -> QTrainStepStats:
        self._validate_batch(batch)

        selected_taps = self._compute_selected_action_taps(batch)
        q_pred = self.q_network(
            goal_position=batch.obs["goal_position"],
            current_velocity=batch.obs["current_velocity"],
            planned_velocities=batch.actions,
            tapped_future_features=selected_taps,
        )

        with torch.no_grad():
            target, next_q_selected, selection_entropy = self._compute_td_target(batch)

        loss = self._compute_loss(q_pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.optimizer.step()
        soft_update_module(self.q_network, self.target_q_network, tau=float(self.config.target_tau))

        return QTrainStepStats(
            loss=float(loss.detach().item()),
            q_pred_mean=float(q_pred.detach().mean().item()),
            target_mean=float(target.detach().mean().item()),
            next_q_mean=float(next_q_selected.detach().mean().item()),
            selection_entropy_mean=float(selection_entropy.detach().mean().item()),
            reward_mean=float(batch.rewards.detach().mean().item()),
            done_fraction=float(batch.dones.detach().float().mean().item()),
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
        selection_indices, selection_probs = sample_action_indices_from_q_scores(
            next_q_scores,
            temperature=float(self.config.selection_temperature),
            generator=self._get_selection_rng(next_q_scores.device),
        )
        next_q_selected = next_q_scores[
            torch.arange(batch_size, device=next_q_scores.device),
            selection_indices,
        ]
        done_mask = 1.0 - batch.dones.to(dtype=torch.float32)
        target = batch.rewards + float(self.config.discount) * done_mask * next_q_selected
        selection_entropy = -torch.sum(selection_probs * torch.log(selection_probs.clamp_min(1e-8)), dim=1)
        return target, next_q_selected, selection_entropy

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

    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "mse":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target)

    def _get_selection_rng(self, device: torch.device) -> torch.Generator:
        if self._selection_rng is None or self._selection_rng_device != device:
            self._selection_rng = torch.Generator(device=device)
            self._selection_rng.manual_seed(int(self.config.selection_seed))
            self._selection_rng_device = device
        return self._selection_rng

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