from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.q_common import compute_td_target_from_next_q_scores, soft_update_module
from ..replay_buffer import ReplaySampleBatch


def validate_common_random_candidate_q_config(config: object) -> None:
    discount = float(getattr(config, "discount"))
    target_tau = float(getattr(config, "target_tau"))
    selection_temperature = float(getattr(config, "selection_temperature"))
    num_bootstrap_candidates = int(getattr(config, "num_bootstrap_candidates"))
    max_speed = float(getattr(config, "max_speed"))
    delta_std = float(getattr(config, "delta_std"))
    dt = float(getattr(config, "dt"))
    grad_clip_norm = getattr(config, "grad_clip_norm")
    loss_type = str(getattr(config, "loss_type"))

    if not (0.0 <= discount <= 1.0):
        raise ValueError("discount must be in [0, 1]")
    if not (0.0 <= target_tau <= 1.0):
        raise ValueError("target_tau must be in [0, 1]")
    if selection_temperature <= 0.0:
        raise ValueError("selection_temperature must be > 0")
    if num_bootstrap_candidates <= 0:
        raise ValueError("num_bootstrap_candidates must be > 0")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be > 0")
    if delta_std < 0.0:
        raise ValueError("delta_std must be >= 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when provided")
    if loss_type not in {"mse", "smooth_l1"}:
        raise ValueError("loss_type must be 'mse' or 'smooth_l1'")


class BaseQTrainer(ABC):
    def __init__(
        self,
        *,
        q_network: nn.Module,
        target_q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: object,
    ) -> None:
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.optimizer = optimizer
        self.config = config
        self._selection_rng: torch.Generator | None = None
        self._selection_rng_device: torch.device | None = None

    def train_step(self, batch: ReplaySampleBatch):
        self._validate_batch(batch)
        q_pred = self._compute_q_pred(batch)
        with torch.no_grad():
            target, next_q_selected, selection_entropy = self._compute_td_target(batch)

        loss = self._compute_loss(q_pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=float(self.config.grad_clip_norm))
        self.optimizer.step()

        return self._build_train_step_stats(
            loss=float(loss.detach().item()),
            q_pred_mean=float(q_pred.detach().mean().item()),
            target_mean=float(target.detach().mean().item()),
            next_q_mean=float(next_q_selected.detach().mean().item()),
            selection_entropy_mean=float(selection_entropy.detach().mean().item()),
            reward_mean=float(batch.rewards.detach().mean().item()),
            done_fraction=float(batch.dones.detach().float().mean().item()),
        )

    def update_target_network(self) -> None:
        soft_update_module(self.q_network, self.target_q_network, tau=float(self.config.target_tau))

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

    def _build_td_target_from_next_q_scores(
        self,
        batch: ReplaySampleBatch,
        next_q_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_td_target_from_next_q_scores(
            rewards=batch.rewards,
            dones=batch.dones,
            next_q_scores=next_q_scores,
            discount=float(self.config.discount),
            temperature=float(self.config.selection_temperature),
            generator=self._get_selection_rng(next_q_scores.device),
        )

    @abstractmethod
    def _validate_batch(self, batch: ReplaySampleBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def _compute_q_pred(self, batch: ReplaySampleBatch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _compute_td_target(self, batch: ReplaySampleBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _build_train_step_stats(
        self,
        *,
        loss: float,
        q_pred_mean: float,
        target_mean: float,
        next_q_mean: float,
        selection_entropy_mean: float,
        reward_mean: float,
        done_fraction: float,
    ):
        raise NotImplementedError


# Backward compatibility alias for older imports.
BaseRandomCandidateQTrainer = BaseQTrainer
