from __future__ import annotations

import math
from typing import Iterable, Sequence

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


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


def _extract_features(
    states: torch.Tensor | dict[str, torch.Tensor],
    tap_projector: nn.Module | None = None,
) -> torch.Tensor:
    if tap_projector is not None:
        return tap_projector(states)
    return _flatten_states(states)


class _TapBottleneckFeatureProjector(nn.Module):
    """Project precomputed decoder tap features and concatenate vector terms."""

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        tap_feature_key: str,
        tap_bottleneck_hidden_dims: tuple[int, ...],
        tap_bottleneck_output_dim: int,
    ) -> None:
        super().__init__()

        self.enabled = False
        self.tap_feature_key = str(tap_feature_key)
        self.features_dim = 0
        self.vector_keys: tuple[str, ...] = ()
        self._flat_key_order: tuple[str, ...] = ()
        self._key_shapes: dict[str, tuple[int, ...]] = {}
        self._key_sizes: dict[str, int] = {}
        self.tap_bottleneck = None

        if not isinstance(observation_space, gym.spaces.Dict):
            return
        if self.tap_feature_key not in observation_space.spaces:
            return

        hidden_dims = tuple(int(v) for v in tap_bottleneck_hidden_dims)
        if not hidden_dims:
            raise ValueError("tap_bottleneck_hidden_dims must contain at least one value")
        if any(v <= 0 for v in hidden_dims):
            raise ValueError("tap_bottleneck_hidden_dims values must be > 0")
        if int(tap_bottleneck_output_dim) <= 0:
            raise ValueError("tap_bottleneck_output_dim must be > 0")

        self.enabled = True
        self._flat_key_order = tuple(sorted(observation_space.spaces.keys()))
        self._key_shapes = {
            key: tuple(int(v) for v in observation_space.spaces[key].shape)
            for key in self._flat_key_order
        }
        self._key_sizes = {
            key: _product(self._key_shapes[key])
            for key in self._flat_key_order
        }
        self.vector_keys = tuple(key for key in self._flat_key_order if key != self.tap_feature_key)

        tap_input_dim = _product(self._key_shapes[self.tap_feature_key])
        self.tap_bottleneck = _build_mlp(
            input_dim=tap_input_dim,
            output_dim=int(tap_bottleneck_output_dim),
            hidden_dims=hidden_dims,
        )
        vector_dim = sum(_product(self._key_shapes[key]) for key in self.vector_keys)
        self.features_dim = int(vector_dim + int(tap_bottleneck_output_dim))

    def _as_obs_dict_from_dict(self, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def _as_obs_dict_from_flat(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
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

    def _as_obs_dict(self, states: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(states, dict):
            return self._as_obs_dict_from_dict(states)
        return self._as_obs_dict_from_flat(states)

    def forward(self, states: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.enabled or self.tap_bottleneck is None:
            raise RuntimeError("Tap bottleneck projector is not enabled")

        observations = self._as_obs_dict(states)
        tap_input = observations[self.tap_feature_key].float().flatten(start_dim=1)
        tap_features = self.tap_bottleneck(tap_input)

        if self.vector_keys:
            vector_features = torch.cat(
                [observations[key].float().flatten(start_dim=1) for key in self.vector_keys],
                dim=1,
            )
            return torch.cat([vector_features, tap_features], dim=1)
        return tap_features


def build_tap_bottleneck_feature_projector(
    observation_space: gym.Space,
    *,
    tap_feature_key: str = "decoder_tap",
    tap_bottleneck_hidden_dims: tuple[int, ...] = (128,),
    tap_bottleneck_output_dim: int = 32,
) -> _TapBottleneckFeatureProjector:
    return _TapBottleneckFeatureProjector(
        observation_space,
        tap_feature_key=tap_feature_key,
        tap_bottleneck_hidden_dims=tuple(int(v) for v in tap_bottleneck_hidden_dims),
        tap_bottleneck_output_dim=int(tap_bottleneck_output_dim),
    )


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
        tap_bottleneck_hidden_dims: tuple[int, ...] = (128,),
        tap_bottleneck_output_dim: int = 32,
        tap_feature_key: str = "decoder_tap",
        tap_projector: _TapBottleneckFeatureProjector | None = None,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False)

        if float(initial_std) <= 0.0:
            raise ValueError("initial_std must be > 0")
        if float(max_std) <= 1e-6:
            raise ValueError("max_std must be > 1e-6")

        self._min_std = 1e-6
        self._max_std = float(max_std)

        self._tap_projector = tap_projector or build_tap_bottleneck_feature_projector(
            observation_space,
            tap_feature_key=tap_feature_key,
            tap_bottleneck_hidden_dims=tuple(int(v) for v in tap_bottleneck_hidden_dims),
            tap_bottleneck_output_dim=int(tap_bottleneck_output_dim),
        )
        feature_dim = (
            int(self._tap_projector.features_dim)
            if self._tap_projector.enabled
            else int(self.num_observations)
        )

        self.net = _build_mlp(feature_dim, self.num_actions, hidden_dims, final_tanh=True)
        bounded_initial_std = min(max(float(initial_std), self._min_std), self._max_std)
        std_param = math.atanh((bounded_initial_std - self._min_std) / self._max_std)
        self.std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(std_param), dtype=torch.float32)
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        features = _extract_features(
            states,
            self._tap_projector if self._tap_projector.enabled else None,
        )
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
        tap_bottleneck_hidden_dims: tuple[int, ...] = (128,),
        tap_bottleneck_output_dim: int = 32,
        tap_feature_key: str = "decoder_tap",
        tap_projector: _TapBottleneckFeatureProjector | None = None,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self._tap_projector = tap_projector or build_tap_bottleneck_feature_projector(
            observation_space,
            tap_feature_key=tap_feature_key,
            tap_bottleneck_hidden_dims=tuple(int(v) for v in tap_bottleneck_hidden_dims),
            tap_bottleneck_output_dim=int(tap_bottleneck_output_dim),
        )
        feature_dim = (
            int(self._tap_projector.features_dim)
            if self._tap_projector.enabled
            else int(self.num_observations)
        )

        self.net = _build_mlp(feature_dim, 1, hidden_dims)

    def compute(self, inputs, role):
        states = inputs["states"]
        features = _extract_features(
            states,
            self._tap_projector if self._tap_projector.enabled else None,
        )
        return self.net(features), {}


class OccupancyQValueModel(DeterministicMixin, Model):
    """State-action critic for off-policy algorithms such as SAC."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device,
        hidden_dims: tuple[int, ...] = (256, 256),
        tap_bottleneck_hidden_dims: tuple[int, ...] = (128,),
        tap_bottleneck_output_dim: int = 32,
        tap_feature_key: str = "decoder_tap",
        tap_projector: _TapBottleneckFeatureProjector | None = None,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self._tap_projector = tap_projector or build_tap_bottleneck_feature_projector(
            observation_space,
            tap_feature_key=tap_feature_key,
            tap_bottleneck_hidden_dims=tuple(int(v) for v in tap_bottleneck_hidden_dims),
            tap_bottleneck_output_dim=int(tap_bottleneck_output_dim),
        )
        feature_dim = (
            int(self._tap_projector.features_dim)
            if self._tap_projector.enabled
            else int(self.num_observations)
        )
        self.net = _build_mlp(feature_dim + self.num_actions, 1, hidden_dims)

    def compute(self, inputs, role):
        states = inputs["states"]
        taken_actions = inputs["taken_actions"]

        features = _extract_features(
            states,
            self._tap_projector if self._tap_projector.enabled else None,
        )
        if not torch.is_tensor(taken_actions):
            taken_actions = torch.as_tensor(taken_actions)
        if taken_actions.ndim == 1:
            taken_actions = taken_actions.unsqueeze(0)
        taken_actions = taken_actions.float()

        q_input = torch.cat((features, taken_actions), dim=1)
        return self.net(q_input), {}
