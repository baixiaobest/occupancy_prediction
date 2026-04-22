from __future__ import annotations

import torch
import torch.nn as nn


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


def compute_td_target_from_next_q_scores(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_scores: torch.Tensor,
    discount: float,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selection_indices, selection_probs = sample_action_indices_from_q_scores(
        next_q_scores,
        temperature=float(temperature),
        generator=generator,
    )
    batch_size = int(next_q_scores.shape[0])
    next_q_selected = next_q_scores[
        torch.arange(batch_size, device=next_q_scores.device),
        selection_indices,
    ]
    done_mask = 1.0 - torch.as_tensor(dones, dtype=torch.float32, device=next_q_scores.device)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=next_q_scores.device)
    target = rewards_tensor + float(discount) * done_mask * next_q_selected
    selection_entropy = -torch.sum(selection_probs * torch.log(selection_probs.clamp_min(1e-8)), dim=1)
    return target, next_q_selected, selection_entropy