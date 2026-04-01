from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute analytic KL divergence between N(mu, sigma^2) and N(0, I)."""
    var = sigma.pow(2)
    kl = 0.5 * (mu.pow(2) + var - torch.log(var + 1e-8) - 1)
    kl = kl.sum(dim=(1, 2, 3, 4)).mean()
    return kl


def kl_target_loss(
    kl_value: torch.Tensor,
    target_kl: float,
) -> torch.Tensor:
    """Quadratic KL target penalty: (KL - target_kl)^2."""
    target = torch.as_tensor(float(target_kl), device=kl_value.device, dtype=kl_value.dtype)
    return (kl_value - target).pow(2)


def weighted_bernoulli_recon_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    occupied_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute occupied-weighted binary cross-entropy."""
    if occupied_weight < 1.0:
        raise ValueError("occupied_weight must be >= 1.0")
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be one of: 'mean', 'none'")

    pos_w = torch.tensor(float(occupied_weight), device=target.device, dtype=target.dtype)
    weights = torch.where(target > 0.5, pos_w, torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction=reduction)


def weighted_focal_recon_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    occupied_weight: float,
    focal_gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute occupied-weighted focal BCE between logits and target."""
    if occupied_weight < 1.0:
        raise ValueError("occupied_weight must be >= 1.0")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be >= 0.0")
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be one of: 'mean', 'none'")

    bce_per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    focal_factor = (1.0 - p_t).pow(focal_gamma)

    pos_w = torch.tensor(float(occupied_weight), device=target.device, dtype=target.dtype)
    weights = torch.where(target > 0.5, pos_w, torch.ones_like(target))
    loss = weights * focal_factor * bce_per_cell
    if reduction == "none":
        return loss
    return loss.mean()


def bernoulli_entropy_loss(
    logits: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Bernoulli entropy from logits.

    This is the per-cell predictive entropy:
      H(p) = -p log p - (1-p) log(1-p), where p = sigmoid(logits).
    """
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be one of: 'mean', 'none'")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    if reduction == "none":
        return entropy
    return entropy.mean()
