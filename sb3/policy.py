from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


def _product(shape: Sequence[int]) -> int:
    out = 1
    for v in shape:
        out *= int(v)
    return int(out)


class OccupancyFusionFeaturesExtractor(BaseFeaturesExtractor):
    """Fuse map observations and vector observations into one latent tensor.

    Expected dict observation keys by default:
    - dynamic_context: dynamic occupancy context maps.
    - static_map: static occupancy map.
    - all remaining keys are flattened and treated as vector features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        *,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        conv_channels: Sequence[int] = (16, 32, 64),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        activation_fn: type[torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("OccupancyFusionFeaturesExtractor requires a Dict observation space")
        if dynamic_key not in observation_space.spaces:
            raise ValueError(f"Missing dynamic occupancy key '{dynamic_key}' in observation space")
        if static_key not in observation_space.spaces:
            raise ValueError(f"Missing static occupancy key '{static_key}' in observation space")

        self.dynamic_key = str(dynamic_key)
        self.static_key = str(static_key)
        self.vector_keys = tuple(
            key for key in observation_space.spaces.keys() if key not in {self.dynamic_key, self.static_key}
        )

        dynamic_shape = tuple(int(v) for v in observation_space.spaces[self.dynamic_key].shape)
        static_shape = tuple(int(v) for v in observation_space.spaces[self.static_key].shape)

        dynamic_channels, dynamic_t, dynamic_h, dynamic_w = self._infer_dynamic_shape(dynamic_shape)
        static_channels, static_t, static_h, static_w = self._infer_static_shape(static_shape)
        if dynamic_h != static_h or dynamic_w != static_w:
            raise ValueError(
                "dynamic and static occupancy maps must have matching HxW, "
                f"got dynamic=({dynamic_h}, {dynamic_w}) and static=({static_h}, {static_w})"
            )
        if static_t not in (1, dynamic_t):
            raise ValueError(
                "static occupancy time dimension must be 1 or match dynamic horizon, "
                f"got static_t={static_t}, dynamic_t={dynamic_t}"
            )

        self.dynamic_channels = int(dynamic_channels)
        self.static_channels = int(static_channels)
        self.dynamic_time_steps = int(dynamic_t)

        # Pack (C, T) into channels for Conv2D, mirroring VAE time-to-channel conditioning.
        static_effective_t = int(dynamic_t)
        map_in_channels = int((self.dynamic_channels * self.dynamic_time_steps) + (self.static_channels * static_effective_t))
        vector_dim = sum(
            _product(tuple(int(v) for v in observation_space.spaces[key].shape)) for key in self.vector_keys
        )

        super().__init__(observation_space=observation_space, features_dim=1)

        conv_layers: list[torch.nn.Module] = []
        in_ch = map_in_channels
        for out_ch in conv_channels:
            out_dim = int(out_ch)
            if out_dim <= 0:
                raise ValueError("conv channels must be positive")
            conv_layers.append(
                torch.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_dim,
                    kernel_size=int(conv_kernel_size),
                    stride=int(conv_stride),
                    padding=int(conv_padding),
                )
            )
            conv_layers.append(activation_fn())
            in_ch = out_dim
        self.map_encoder = torch.nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, map_in_channels, dynamic_h, dynamic_w)
            encoded = self.map_encoder(dummy)
            map_latent_dim = int(encoded.flatten(start_dim=1).shape[1])

        self.map_height = int(dynamic_h)
        self.map_width = int(dynamic_w)
        self._features_dim = int(vector_dim + map_latent_dim)

    @staticmethod
    def _infer_dynamic_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        # Dynamic occupancy is expected as (C, T, H, W) in observation space.
        if len(shape) != 4:
            raise ValueError(
                "dynamic occupancy shape must be (C, T, H, W), "
                f"got shape={shape}"
            )
        c, t, h, w = shape
        if int(c) <= 0 or int(t) <= 0 or int(h) <= 0 or int(w) <= 0:
            raise ValueError("dynamic occupancy dimensions must be positive")
        return int(c), int(t), int(h), int(w)

    @staticmethod
    def _infer_static_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        # Static occupancy can be (C, H, W) or (C, T, H, W).
        if len(shape) == 3:
            c, h, w = shape
            if int(c) <= 0 or int(h) <= 0 or int(w) <= 0:
                raise ValueError("static occupancy dimensions must be positive")
            return int(c), 1, int(h), int(w)
        if len(shape) == 4:
            c, t, h, w = shape
            if int(c) <= 0 or int(t) <= 0 or int(h) <= 0 or int(w) <= 0:
                raise ValueError("static occupancy dimensions must be positive")
            return int(c), int(t), int(h), int(w)
        raise ValueError(
            "static occupancy shape must be (C, H, W) or (C, T, H, W), "
            f"got shape={shape}"
        )

    def _normalize_dynamic_context(self, x: torch.Tensor) -> torch.Tensor:
        # VAE-aligned: dynamic_context is (B, C, T, H, W).
        if x.ndim != 5:
            raise ValueError(
                "dynamic_context must have shape (B, C, T, H, W), "
                f"got {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.dynamic_channels:
            raise ValueError(
                f"dynamic_context channel dimension must be {self.dynamic_channels}, got {int(x.shape[1])}"
            )
        if int(x.shape[2]) != self.dynamic_time_steps:
            raise ValueError(
                f"dynamic_context time dimension must be {self.dynamic_time_steps}, got {int(x.shape[2])}"
            )
        if tuple(int(v) for v in x.shape[-2:]) != (self.map_height, self.map_width):
            raise ValueError(
                "dynamic_context HxW mismatch: expected "
                f"({self.map_height}, {self.map_width}), got {tuple(int(v) for v in x.shape[-2:])}"
            )
        return x.float()

    def _normalize_static_context(self, x: torch.Tensor, *, target_t: int) -> torch.Tensor:
        # Static can be (B, C, H, W) or (B, C, T, H, W).
        if x.ndim == 4:
            x = x.unsqueeze(2)
        elif x.ndim != 5:
            raise ValueError(
                "static_map must have shape (B, C, H, W) or (B, C, T, H, W), "
                f"got {tuple(x.shape)}"
            )

        if int(x.shape[1]) != self.static_channels:
            raise ValueError(
                f"static_map channel dimension must be {self.static_channels}, got {int(x.shape[1])}"
            )
        if tuple(int(v) for v in x.shape[-2:]) != (self.map_height, self.map_width):
            raise ValueError(
                "static_map HxW mismatch: expected "
                f"({self.map_height}, {self.map_width}), got {tuple(int(v) for v in x.shape[-2:])}"
            )

        static_t = int(x.shape[2])
        if static_t == 1 and target_t > 1:
            x = x.expand(-1, -1, target_t, -1, -1)
        elif static_t != target_t:
            raise ValueError(
                f"static_map time dimension must be 1 or {target_t}, got {static_t}"
            )
        return x.float()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        dynamic = self._normalize_dynamic_context(observations[self.dynamic_key])
        static = self._normalize_static_context(observations[self.static_key], target_t=int(dynamic.shape[2]))

        b, c_dyn, t_ctx, h, w = dynamic.shape
        _, c_static, _, _, _ = static.shape
        dynamic_2d = dynamic.reshape(b, c_dyn * t_ctx, h, w)
        static_2d = static.reshape(b, c_static * t_ctx, h, w)
        map_input = torch.cat([static_2d, dynamic_2d], dim=1)
        map_latent = self.map_encoder(map_input).flatten(start_dim=1)

        if self.vector_keys:
            vector_features = torch.cat(
                [observations[key].float().flatten(start_dim=1) for key in self.vector_keys],
                dim=1,
            )
            return torch.cat([vector_features, map_latent], dim=1)
        return map_latent


class OccupancyActorNetwork(torch.nn.Module):
    """Actor MLP over fused [vector features, encoded occupancy map]."""

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


class OccupancyCriticNetwork(torch.nn.Module):
    """Critic MLP over fused [vector features, encoded occupancy map]."""

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


class OccupancyExtractor(torch.nn.Module):
    """Actor-critic extractor composed of explicit actor and critic networks."""

    def __init__(
        self,
        *,
        feature_dim: int,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        actor_activation_fn: type[torch.nn.Module],
        critic_activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.policy_net = OccupancyActorNetwork(
            input_dim=feature_dim,
            hidden_dims=actor_hidden_dims,
            activation_fn=actor_activation_fn,
        )
        self.value_net = OccupancyCriticNetwork(
            input_dim=feature_dim,
            hidden_dims=critic_hidden_dims,
            activation_fn=critic_activation_fn,
        )
        self.latent_dim_pi = self.policy_net.output_dim
        self.latent_dim_vf = self.value_net.output_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class OccupancyActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with occupancy-map fusion + explicit actor/critic MLP heads."""

    def __init__(
        self,
        *args,
        actor_hidden_dims: Sequence[int] = (64, 64),
        critic_hidden_dims: Sequence[int] = (64, 64),
        actor_activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        critic_activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        map_conv_channels: Sequence[int] = (16, 32, 64),
        map_conv_kernel_size: int = 3,
        map_conv_stride: int = 2,
        map_conv_padding: int = 1,
        map_activation_fn: type[torch.nn.Module] = torch.nn.ReLU,
        **kwargs,
    ) -> None:
        self.actor_hidden_dims = tuple(int(dim) for dim in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(dim) for dim in critic_hidden_dims)
        self.actor_activation_fn = actor_activation_fn
        self.critic_activation_fn = critic_activation_fn

        kwargs.setdefault("ortho_init", True)
        kwargs.setdefault("share_features_extractor", True)
        kwargs.setdefault("features_extractor_class", OccupancyFusionFeaturesExtractor)

        features_extractor_kwargs = dict(kwargs.pop("features_extractor_kwargs", {}))
        features_extractor_kwargs.setdefault("dynamic_key", str(dynamic_key))
        features_extractor_kwargs.setdefault("static_key", str(static_key))
        features_extractor_kwargs.setdefault("conv_channels", tuple(int(v) for v in map_conv_channels))
        features_extractor_kwargs.setdefault("conv_kernel_size", int(map_conv_kernel_size))
        features_extractor_kwargs.setdefault("conv_stride", int(map_conv_stride))
        features_extractor_kwargs.setdefault("conv_padding", int(map_conv_padding))
        features_extractor_kwargs.setdefault("activation_fn", map_activation_fn)
        kwargs["features_extractor_kwargs"] = features_extractor_kwargs

        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = OccupancyExtractor(
            feature_dim=self.features_dim,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            actor_activation_fn=self.actor_activation_fn,
            critic_activation_fn=self.critic_activation_fn,
        )


__all__ = [
    "OccupancyActorCriticPolicy",
    "OccupancyActorNetwork",
    "OccupancyCriticNetwork",
    "OccupancyExtractor",
    "OccupancyFusionFeaturesExtractor",
]
