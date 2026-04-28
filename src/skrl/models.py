from __future__ import annotations

import math
from typing import Iterable, Sequence

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from src.vae_decoder_tap_utils import (
    _normalize_dynamic_context_tensor,
    _normalize_static_context_tensor,
    infer_dynamic_occupancy_shape,
    infer_static_occupancy_shape,
    load_vae_decoder_tap_bundle,
    state_to_xy_vec,
)


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int], final_tanh: bool = False) -> nn.Sequential:
    dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _product(shape: Sequence[int]) -> int:
    out = 1
    for value in shape:
        out *= int(value)
    return int(out)


def _flatten_states(states: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(states, dict):
        pieces: list[torch.Tensor] = []
        for key in sorted(states.keys()):
            value = states[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if value.ndim == 0:
                value = value.view(1, 1)
            elif value.ndim == 1:
                value = value.unsqueeze(0)
            pieces.append(value.float().flatten(start_dim=1))
        if not pieces:
            raise ValueError("states dict is empty")
        return torch.cat(pieces, dim=1)

    if not torch.is_tensor(states):
        states = torch.as_tensor(states)
    if states.ndim == 0:
        states = states.view(1, 1)
    elif states.ndim == 1:
        states = states.unsqueeze(0)
    return states.float().flatten(start_dim=1)


def _resolve_feature_dim(feature_extractor: nn.Module | None, num_observations: int) -> int:
    if feature_extractor is None:
        return int(num_observations)
    return int(getattr(feature_extractor, "features_dim"))


def _extract_features(
    states: torch.Tensor | dict[str, torch.Tensor],
    feature_extractor: nn.Module | None,
) -> torch.Tensor:
    if feature_extractor is not None:
        return feature_extractor(states)
    return _flatten_states(states)


class VAEDecoderTapFeatureExtractor(nn.Module):
    """Frozen feature extractor that taps latent features from a VAE decoder."""

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        vae_checkpoint: str,
        tap_layer: int | None = None,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        velocity_key: str = "current_velocity",
        position_key: str = "goal_position",
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("VAEDecoderTapFeatureExtractor requires a Dict observation space")

        self.observation_space = observation_space
        self.dynamic_key = str(dynamic_key)
        self.static_key = str(static_key)
        self.velocity_key = str(velocity_key)
        self.position_key = str(position_key)

        required_keys = [self.dynamic_key, self.static_key, self.velocity_key, self.position_key]
        for key in required_keys:
            if key not in observation_space.spaces:
                raise ValueError(f"Missing key '{key}' in observation space")

        self._flat_key_order = tuple(sorted(observation_space.spaces.keys()))
        self._key_shapes = {
            key: tuple(int(v) for v in observation_space.spaces[key].shape)
            for key in self._flat_key_order
        }
        self._key_sizes = {
            key: _product(self._key_shapes[key])
            for key in self._flat_key_order
        }
        self.vector_keys = tuple(
            key for key in self._flat_key_order if key not in {self.dynamic_key, self.static_key}
        )

        dynamic_shape = tuple(int(v) for v in observation_space.spaces[self.dynamic_key].shape)
        static_shape = tuple(int(v) for v in observation_space.spaces[self.static_key].shape)
        dynamic_channels, dynamic_t, dynamic_h, dynamic_w = infer_dynamic_occupancy_shape(dynamic_shape)
        static_channels, static_t, static_h, static_w = infer_static_occupancy_shape(static_shape)
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
        self.map_height = int(dynamic_h)
        self.map_width = int(dynamic_w)

        vector_dim = sum(
            _product(tuple(int(v) for v in observation_space.spaces[key].shape)) for key in self.vector_keys
        )

        tap_bundle = load_vae_decoder_tap_bundle(vae_checkpoint)
        self.decoder = tap_bundle.decoder
        self.decoder_context_frames = int(tap_bundle.decoder_context_frames)
        self.latent_channels = int(tap_bundle.latent_channels)
        self.latent_height = int(tap_bundle.latent_height)
        self.latent_width = int(tap_bundle.latent_width)
        num_upsample_blocks = int(tap_bundle.num_upsample_blocks)

        self.register_buffer(
            "fixed_z",
            torch.randn(
                (1, self.latent_channels, 1, self.latent_height, self.latent_width),
                dtype=torch.float32,
            ),
            persistent=True,
        )

        if tap_layer is None:
            self.tap_layer = int(num_upsample_blocks - 1)
        else:
            self.tap_layer = int(tap_layer)

        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()

        with torch.no_grad():
            dynamic_dummy = torch.zeros((1,) + dynamic_shape, dtype=torch.float32)
            static_dummy = torch.zeros((1,) + static_shape, dtype=torch.float32)
            dynamic_dummy = _normalize_dynamic_context_tensor(
                dynamic_dummy,
                expected_channels=self.dynamic_channels,
                expected_time=self.dynamic_time_steps,
                expected_hw=(self.map_height, self.map_width),
            )
            static_dummy = _normalize_static_context_tensor(
                static_dummy,
                expected_channels=self.static_channels,
                expected_hw=(self.map_height, self.map_width),
                target_t=self.dynamic_time_steps,
            )
            velocity_dummy = state_to_xy_vec(
                torch.zeros((1,) + self._key_shapes[self.velocity_key], dtype=torch.float32),
                key_name=self.velocity_key,
            )
            position_dummy = state_to_xy_vec(
                torch.zeros((1,) + self._key_shapes[self.position_key], dtype=torch.float32),
                key_name=self.position_key,
            )
            tapped = self.decoder(
                self.fixed_z,
                dynamic_dummy,
                static_dummy,
                velocity_dummy,
                position_dummy,
                tap_layer=self.tap_layer,
                tap_only=True,
            )
            map_latent_dim = int(tapped.flatten(start_dim=1).shape[1])

        self.features_dim = int(vector_dim + map_latent_dim)

    def train(self, mode: bool = True) -> VAEDecoderTapFeatureExtractor:
        super().train(mode)
        self.decoder.eval()
        return self

    def _as_obs_dict(self, states: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(states, dict):
            out: dict[str, torch.Tensor] = {}
            batch_size: int | None = None
            for key in self._flat_key_order:
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states dict")
                value = states[key]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                if value.ndim == len(self._key_shapes[key]):
                    value = value.unsqueeze(0)
                expected_ndim = len(self._key_shapes[key]) + 1
                if value.ndim != expected_ndim:
                    raise ValueError(
                        f"states['{key}'] must have ndim={expected_ndim}, got {value.ndim}"
                    )
                if tuple(int(v) for v in value.shape[1:]) != self._key_shapes[key]:
                    raise ValueError(
                        f"states['{key}'] shape mismatch: expected {self._key_shapes[key]}, "
                        f"got {tuple(int(v) for v in value.shape[1:])}"
                    )
                if batch_size is None:
                    batch_size = int(value.shape[0])
                elif int(value.shape[0]) != batch_size:
                    raise ValueError("all states dict tensors must have the same batch dimension")
                out[key] = value.float()
            return out

        flat = _flatten_states(states)
        out = {}
        start = 0
        for key in self._flat_key_order:
            end = start + self._key_sizes[key]
            if end > int(flat.shape[1]):
                raise ValueError(
                    f"flat states dimension too small: expected at least {end}, got {int(flat.shape[1])}"
                )
            out[key] = flat[:, start:end].reshape(flat.shape[0], *self._key_shapes[key])
            start = end
        if start != int(flat.shape[1]):
            raise ValueError(
                f"flat states dimension mismatch: expected {start}, got {int(flat.shape[1])}"
            )
        return out

    def forward(self, states: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        observations = self._as_obs_dict(states)
        dynamic = _normalize_dynamic_context_tensor(
            observations[self.dynamic_key],
            expected_channels=self.dynamic_channels,
            expected_time=self.dynamic_time_steps,
            expected_hw=(self.map_height, self.map_width),
        )
        static = _normalize_static_context_tensor(
            observations[self.static_key],
            expected_channels=self.static_channels,
            expected_hw=(self.map_height, self.map_width),
            target_t=int(dynamic.shape[2]),
        )
        velocity_vec = state_to_xy_vec(observations[self.velocity_key], key_name=self.velocity_key)
        position_vec = state_to_xy_vec(observations[self.position_key], key_name=self.position_key)

        z = self.fixed_z.expand(dynamic.shape[0], -1, -1, -1, -1).to(
            device=dynamic.device,
            dtype=dynamic.dtype,
        )

        with torch.no_grad():
            tapped = self.decoder(
                z,
                dynamic,
                static,
                velocity_vec,
                position_vec,
                tap_layer=self.tap_layer,
                tap_only=True,
            )
            map_latent = tapped.flatten(start_dim=1)

        if self.vector_keys:
            vector_features = torch.cat(
                [observations[key].float().flatten(start_dim=1) for key in self.vector_keys],
                dim=1,
            )
            return torch.cat([vector_features, map_latent], dim=1)
        return map_latent


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
        feature_extractor: nn.Module | None = None,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False)

        if float(initial_std) <= 0.0:
            raise ValueError("initial_std must be > 0")
        if float(max_std) <= 1e-6:
            raise ValueError("max_std must be > 1e-6")

        self._min_std = 1e-6
        self._max_std = float(max_std)

        self.feature_extractor = feature_extractor
        feature_dim = _resolve_feature_dim(self.feature_extractor, self.num_observations)

        self.net = _build_mlp(feature_dim, self.num_actions, hidden_dims, final_tanh=True)
        bounded_initial_std = min(max(float(initial_std), self._min_std), self._max_std)
        std_param = math.atanh((bounded_initial_std - self._min_std) / self._max_std)
        self.std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(std_param), dtype=torch.float32)
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        features = _extract_features(states, self.feature_extractor)
        return self.net(features), self._squash_std_parameter(), {}

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
        feature_extractor: nn.Module | None = None,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.feature_extractor = feature_extractor
        feature_dim = _resolve_feature_dim(self.feature_extractor, self.num_observations)

        self.net = _build_mlp(feature_dim, 1, hidden_dims)

    def compute(self, inputs, role):
        states = inputs["states"]
        features = _extract_features(states, self.feature_extractor)
        return self.net(features), {}
