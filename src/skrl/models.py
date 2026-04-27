from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int]) -> nn.Sequential:
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class OccupancyPolicyModel(GaussianMixin, Model):
    """Simple Gaussian actor over flattened observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0)

        self.net = _build_mlp(self.num_observations, self.num_actions, hidden_dims)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -0.5, dtype=torch.float32))

    def compute(self, inputs, role):
        states = inputs["states"].float().view(inputs["states"].shape[0], -1)
        return self.net(states), self.log_std_parameter, {}


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
