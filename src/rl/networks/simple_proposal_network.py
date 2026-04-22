from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Module, int]:
    dims = [int(dim) for dim in hidden_dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dims must contain positive integers")

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.ReLU(inplace=True))
        last_dim = dim
    mlp: nn.Module = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
    return mlp, last_dim


class SimpleVelocityProposalNetwork(nn.Module):
    """Predict horizon delta-velocity distribution from simple state vectors.

    Inputs:
    - current_velocity: (B, 2)
    - goal_relative_position: (B, 2)

    Outputs:
    - delta_velocity_mean: (B, horizon, 2)
    - delta_velocity_variance: (B, horizon, 2), strictly positive
    """

    def __init__(
        self,
        horizon: int,
        hidden_dims: Sequence[int] = (128, 128),
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__()

        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.min_variance = float(min_variance)
        if self.min_variance <= 0.0:
            raise ValueError("min_variance must be > 0")

        self.mlp, mlp_output_dim = _build_mlp(input_dim=4, hidden_dims=hidden_dims)
        self.delta_velocity_mean_head = nn.Linear(mlp_output_dim, self.horizon * 2)
        self.delta_velocity_var_head = nn.Linear(mlp_output_dim, self.horizon * 2)

    @staticmethod
    def _prepare_input(value: torch.Tensor, *, name: str, device: torch.device | None = None) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"{name} must have shape (B, 2)")
        return tensor

    def forward(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_relative_position: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        goal_input = goal_relative_position if goal_relative_position is not None else goal_position
        if goal_input is None:
            raise ValueError("goal_relative_position must be provided")

        velocity = self._prepare_input(current_velocity, name="current_velocity")
        goal = self._prepare_input(goal_input, name="goal_relative_position", device=velocity.device)

        features = self.mlp(torch.cat([velocity, goal], dim=1))
        batch_size = int(features.shape[0])
        delta_velocity_mean = self.delta_velocity_mean_head(features).view(batch_size, self.horizon, 2)
        delta_velocity_variance = (
            F.softplus(self.delta_velocity_var_head(features)).view(batch_size, self.horizon, 2)
            + self.min_variance
        )
        return delta_velocity_mean, delta_velocity_variance

    def sample_delta_velocities(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_relative_position: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
        num_candidates: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        candidates = int(num_candidates)
        if candidates <= 0:
            raise ValueError("num_candidates must be > 0")

        delta_mean, delta_var = self(
            current_velocity=current_velocity,
            goal_relative_position=goal_relative_position,
            goal_position=goal_position,
        )
        batch_size = int(delta_mean.shape[0])
        eps = torch.randn(
            (batch_size, candidates, self.horizon, 2),
            dtype=delta_mean.dtype,
            device=delta_mean.device,
            generator=generator,
        )
        return delta_mean.unsqueeze(1) + torch.sqrt(delta_var).unsqueeze(1) * eps

    def sample_velocity_plans(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_relative_position: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
        num_candidates: int,
        include_current_velocity_candidate: bool = True,
        generator: torch.Generator | None = None,
        max_speed: float | None = None,
    ) -> torch.Tensor:
        velocity = self._prepare_input(current_velocity, name="current_velocity")
        sampled_deltas = self.sample_delta_velocities(
            current_velocity=velocity,
            goal_relative_position=goal_relative_position,
            goal_position=goal_position,
            num_candidates=num_candidates,
            generator=generator,
        )
        plans = velocity.unsqueeze(1).unsqueeze(2) + torch.cumsum(sampled_deltas, dim=2)

        if bool(include_current_velocity_candidate):
            plans[:, 0, :, :] = velocity.unsqueeze(1)

        if max_speed is not None:
            speed_limit = float(max_speed)
            if speed_limit <= 0.0:
                raise ValueError("max_speed must be > 0 when provided")
            plans = self._clip_max_speed(plans, speed_limit)

        return plans

    def sample_actions(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_relative_position: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
        num_candidates: int,
        include_current_velocity_candidate: bool = True,
        generator: torch.Generator | None = None,
        max_speed: float | None = None,
    ) -> torch.Tensor:
        plans = self.sample_velocity_plans(
            current_velocity=current_velocity,
            goal_relative_position=goal_relative_position,
            goal_position=goal_position,
            num_candidates=num_candidates,
            include_current_velocity_candidate=include_current_velocity_candidate,
            generator=generator,
            max_speed=max_speed,
        )
        return plans[:, :, 0, :]

    @staticmethod
    def _clip_max_speed(actions: torch.Tensor, max_speed: float) -> torch.Tensor:
        speed = torch.linalg.vector_norm(actions, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(max_speed / speed, max=1.0)
        return actions * scale


def build_simple_proposal_network(
    horizon: int,
    hidden_dims: Sequence[int] = (128, 128),
    min_variance: float = 1e-6,
    device: torch.device | str | None = None,
) -> SimpleVelocityProposalNetwork:
    model = SimpleVelocityProposalNetwork(
        horizon=horizon,
        hidden_dims=hidden_dims,
        min_variance=min_variance,
    )
    if device is not None:
        model = model.to(device)
    return model


__all__ = ["SimpleVelocityProposalNetwork", "build_simple_proposal_network"]
