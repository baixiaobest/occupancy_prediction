from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    dims = [int(dim) for dim in hidden_dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dims must contain positive integers")

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.ReLU(inplace=True))
        last_dim = dim
    layers.append(nn.Linear(last_dim, 1))
    return nn.Sequential(*layers)


class SimpleStateActionQNetwork(nn.Module):
    """Small MLP Q network over current state and one-step action.

    Inputs:
    - current_velocity: (B, 2)
    - goal_position: (B, 2)  # goal offset in the current observation pipeline
    - action: (B, 2)
    """

    def __init__(self, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.mlp = _build_mlp(input_dim=6, hidden_dims=hidden_dims)

    def forward(
        self,
        *,
        current_velocity: torch.Tensor,
        goal_position: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        current_velocity = self._prepare_input(current_velocity, name="current_velocity")
        goal_position = self._prepare_input(goal_position, name="goal_position")
        action = self._prepare_input(action, name="action")
        features = torch.cat([current_velocity, goal_position, action], dim=1)
        return self.mlp(features).squeeze(-1)

    @staticmethod
    def _prepare_input(value: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"{name} must have shape (B, 2)")
        return tensor


def build_simple_q_network(
    hidden_dims: Sequence[int] = (128, 128),
    device: torch.device | str | None = None,
) -> SimpleStateActionQNetwork:
    model = SimpleStateActionQNetwork(hidden_dims=hidden_dims)
    if device is not None:
        model = model.to(device)
    return model


__all__ = ["SimpleStateActionQNetwork", "build_simple_q_network"]