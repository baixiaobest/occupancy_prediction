from __future__ import annotations

from typing import Sequence

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation_fn: type[torch.nn.Module],
) -> tuple[torch.nn.Module, int]:
    layers: list[torch.nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        dim = int(hidden_dim)
        if dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        layers.append(torch.nn.Linear(last_dim, dim))
        layers.append(activation_fn())
        last_dim = dim

    if not layers:
        return torch.nn.Identity(), int(input_dim)
    return torch.nn.Sequential(*layers), int(last_dim)


class MinimalActorNetwork(torch.nn.Module):
    """Actor MLP from [relative goal, current velocity] to policy latent."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.net, self.output_dim = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MinimalCriticNetwork(torch.nn.Module):
    """Critic MLP from [relative goal, current velocity] to value latent."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.net, self.output_dim = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MinimalActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with explicit actor and critic MLPs."""

    def __init__(
        self,
        *args,
        actor_hidden_dims: Sequence[int] = (64, 64),
        critic_hidden_dims: Sequence[int] = (64, 64),
        **kwargs,
    ) -> None:
        self.actor_hidden_dims = tuple(int(dim) for dim in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(dim) for dim in critic_hidden_dims)
        # Keep initialization simple and stable for this minimal setup.
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            feature_dim=self.features_dim,
            net_arch={
                "pi": list(self.actor_hidden_dims),
                "vf": list(self.critic_hidden_dims),
            },
            activation_fn=self.activation_fn,
            device=self.device,
        )


__all__ = [
    "MinimalActorCriticPolicy",
    "MinimalActorNetwork",
    "MinimalCriticNetwork",
]
