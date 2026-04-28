from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import math

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int], final_tanh: bool=False) -> nn.Sequential:
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class OccupancyPolicyModel(GaussianMixin, Model):
    """Simple Gaussian actor over flattened observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device,
        hidden_dims: tuple[int, ...] = (256, 256),
        initial_std: float = 0.6,
        max_std: float = 1.0,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False)

        if float(initial_std) <= 0.0:
            raise ValueError("initial_std must be > 0")
        if float(max_std) <= 1e-6:
            raise ValueError("max_std must be > 1e-6")

        self._min_std = 1e-6
        self._max_std = float(max_std)

        self.net = _build_mlp(self.num_observations, self.num_actions, hidden_dims, final_tanh=True)
        bounded_initial_std = min(max(float(initial_std), self._min_std), self._max_std)
        std_param = math.atanh((bounded_initial_std - self._min_std) / self._max_std)
        self.std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(std_param), dtype=torch.float32)
        )

    def compute(self, inputs, role):
        states = inputs["states"].float().view(inputs["states"].shape[0], -1)
        return self.net(states), self._squash_std_parameter(), {}

    def _squash_std_parameter(self) -> torch.Tensor:
        # User-facing parameterization: std = tanh(raw_std) * max_std + 1e-6.
        # Clamp avoids invalid negative std values before log conversion required by GaussianMixin.
        std = torch.tanh(self.std_parameter) * self._max_std + self._min_std
        std = torch.clamp(std, min=self._min_std)
        return torch.log(std)


class OccupancyValueModel(DeterministicMixin, Model):
    """Simple critic over flattened observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = _build_mlp(self.num_observations, 1, hidden_dims)

    def compute(self, inputs, role):
        states = inputs["states"].float().view(inputs["states"].shape[0], -1)
        return self.net(states), {}
