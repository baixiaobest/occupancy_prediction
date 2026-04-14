from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_common import (
    HUMAN_WALKING_SPEED_MPS,
    _DownsampleBlock2d,
    _downsample_hw,
    _pack_video_time_to_channel,
    _to_stride2,
)


DEFAULT_PROPOSAL_DOWNSAMPLE_STRIDES: tuple[tuple[int, int], ...] = (
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
)


class _BaseVelocityProposalNetwork(nn.Module):
    def __init__(self, *, horizon: int, min_variance: float) -> None:
        super().__init__()
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.min_variance = float(min_variance)
        if self.min_variance <= 0:
            raise ValueError("min_variance must be > 0")

    @staticmethod
    def _build_mlp_and_heads(
        *,
        input_dim: int,
        mlp_hidden_dims: Sequence[int],
        horizon: int,
    ) -> tuple[nn.Module, nn.Linear, nn.Linear]:
        hidden_dims = [int(d) for d in mlp_hidden_dims]
        if any(d <= 0 for d in hidden_dims):
            raise ValueError("mlp_hidden_dims values must be > 0")

        mlp_layers: list[nn.Module] = []
        head_input_dim = int(input_dim)
        for dim in hidden_dims:
            mlp_layers.append(nn.Linear(head_input_dim, dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            head_input_dim = dim
        mlp = nn.Sequential(*mlp_layers) if len(mlp_layers) > 0 else nn.Identity()

        velocity_mean_head = nn.Linear(head_input_dim, horizon * 2)
        velocity_var_head = nn.Linear(head_input_dim, horizon * 2)
        return mlp, velocity_mean_head, velocity_var_head

    @staticmethod
    def _prepare_vector_input(
        value: torch.Tensor | None,
        *,
        name: str,
        device: torch.device,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if value is None:
            if batch_size is None:
                raise ValueError(f"{name} must be provided when batch_size is unknown")
            return torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

        vec = torch.as_tensor(value, dtype=torch.float32, device=device)
        if vec.ndim == 1:
            if vec.shape[0] != 2:
                raise ValueError(f"{name} 1D input must have shape (2,)")
            vec = vec.unsqueeze(0)
        if vec.ndim != 2 or vec.shape[1] != 2:
            raise ValueError(f"{name} must have shape (B, 2)")
        if batch_size is not None and vec.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must be {batch_size}, got {vec.shape[0]}")
        return vec

    def _predict_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        shared = self.mlp(features)
        velocity_mean = self.velocity_mean_head(shared).view(batch_size, self.horizon, 2)
        velocity_variance = (
            F.softplus(self.velocity_var_head(shared)).view(batch_size, self.horizon, 2) + self.min_variance
        )
        return velocity_mean, velocity_variance

    @staticmethod
    def sample(velocity_mean: torch.Tensor, velocity_variance: torch.Tensor) -> torch.Tensor:
        if velocity_mean.shape != velocity_variance.shape:
            raise ValueError("velocity_mean and velocity_variance must have matching shapes")
        eps = torch.randn_like(velocity_mean)
        return velocity_mean + torch.sqrt(velocity_variance) * eps


class VelocityTrajectoryProposalNetwork(_BaseVelocityProposalNetwork):
    """Convolutional proposal network for future velocity distributions.

    Inputs:
    - dynamic_x: past occupancy sequence with shape (B, C, T, H, W)
    - static_x: static occupancy with shape (B, 1, H, W) or (B, 1, T, H, W)
    - current_velocity: optional current velocity (B, 2)
    - goal_position: optional goal position (B, 2)

    Outputs:
    - velocity_mean: shape (B, horizon, 2)
    - velocity_variance: shape (B, horizon, 2), strictly positive
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        horizon: int,
        channels: Sequence[int] = (32, 64, 128, 128, 128, 128),
        static_stem_channels: int = 8,
        downsample_strides: Sequence[int | Sequence[int]] = DEFAULT_PROPOSAL_DOWNSAMPLE_STRIDES,
        mlp_hidden_dims: Sequence[int] = (256, 256),
        velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon, min_variance=min_variance)

        if len(input_shape) != 4:
            raise ValueError("input_shape must be (C, T, H, W)")

        in_channels = int(input_shape[0])
        self.input_time_steps = int(input_shape[1])
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])
        if self.input_time_steps <= 0 or self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input_shape values must all be > 0")

        self.static_stem_channels = int(static_stem_channels)
        if self.static_stem_channels <= 0:
            raise ValueError("static_stem_channels must be > 0")

        self.velocity_scale = float(velocity_scale)
        if self.velocity_scale <= 0:
            raise ValueError("velocity_scale must be > 0")

        self.channels = [int(c) for c in channels]
        if len(self.channels) < 2:
            raise ValueError("channels must contain at least 2 entries")
        if any(c <= 0 for c in self.channels):
            raise ValueError("all channels values must be > 0")

        if len(downsample_strides) != len(self.channels) - 1:
            raise ValueError("downsample_strides length must equal len(channels)-1")
        stride_list = [_to_stride2(s) for s in downsample_strides]

        self.dynamic_stem = nn.Sequential(
            nn.Conv2d(in_channels * self.input_time_steps, self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.static_stem = nn.Sequential(
            nn.Conv2d(self.input_time_steps, self.static_stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.static_stem_channels),
            nn.ReLU(inplace=True),
        )

        down_blocks: list[nn.Module] = []
        in_ch = self.channels[0] + self.static_stem_channels
        for out_ch, stride in zip(self.channels[1:], stride_list):
            down_blocks.append(_DownsampleBlock2d(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)

        encoded_h, encoded_w = _downsample_hw((self.input_height, self.input_width), stride_list)
        self.encoded_flat_dim = self.channels[-1] * encoded_h * encoded_w

        self.mlp, self.velocity_mean_head, self.velocity_var_head = self._build_mlp_and_heads(
            input_dim=self.encoded_flat_dim + 4,
            mlp_hidden_dims=mlp_hidden_dims,
            horizon=self.horizon,
        )

    def _validate_dynamic_and_static_inputs(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
    ) -> torch.Tensor:
        if dynamic_x.ndim != 5:
            raise ValueError("dynamic_x must have shape (B, C, T, H, W)")
        if dynamic_x.shape[2] != self.input_time_steps:
            raise ValueError(
                f"dynamic_x time dimension must be {self.input_time_steps}, got {dynamic_x.shape[2]}"
            )
        if dynamic_x.shape[3] != self.input_height or dynamic_x.shape[4] != self.input_width:
            raise ValueError(
                "dynamic_x spatial shape does not match model input_shape "
                f"({self.input_height}, {self.input_width})"
            )

        if static_x.ndim == 4:
            static_x = static_x.unsqueeze(2)
        if static_x.ndim != 5:
            raise ValueError("static_x must have shape (B, 1, H, W) or (B, 1, T, H, W)")
        if static_x.shape[1] != 1:
            raise ValueError("static_x channel dimension must be 1")
        if static_x.shape[-2:] != dynamic_x.shape[-2:]:
            raise ValueError("static_x and dynamic_x must share H/W")
        if static_x.shape[2] == 1 and self.input_time_steps > 1:
            static_x = static_x.expand(-1, -1, self.input_time_steps, -1, -1)
        elif static_x.shape[2] != self.input_time_steps:
            raise ValueError("static_x time dimension must be 1 or match dynamic_x")

        return static_x

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
        goal_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        static_x = self._validate_dynamic_and_static_inputs(dynamic_x, static_x)

        batch_size = dynamic_x.shape[0]
        velocity_vec = self._prepare_vector_input(
            current_velocity,
            name="current_velocity",
            device=dynamic_x.device,
            batch_size=batch_size,
        )
        goal_vec = self._prepare_vector_input(
            goal_position,
            name="goal_position",
            device=dynamic_x.device,
            batch_size=batch_size,
        )

        velocity_vec = velocity_vec / self.velocity_scale

        h_dyn = self.dynamic_stem(_pack_video_time_to_channel(dynamic_x))
        h_static = self.static_stem(_pack_video_time_to_channel(static_x))
        h = torch.cat([h_dyn, h_static], dim=1)
        for down_block in self.down_blocks:
            h = down_block(h)

        encoded = torch.flatten(h, start_dim=1)
        mlp_input = torch.cat(
            [
                encoded,
                velocity_vec.to(dtype=encoded.dtype),
                goal_vec.to(dtype=encoded.dtype),
            ],
            dim=1,
        )
        return self._predict_from_features(mlp_input)


class VelocityGoalMLPProposalNetwork(_BaseVelocityProposalNetwork):
    """Simple proposal network using only current velocity and goal position."""

    def __init__(
        self,
        horizon: int,
        mlp_hidden_dims: Sequence[int] = (128, 128),
        velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__(horizon=horizon, min_variance=min_variance)

        self.velocity_scale = float(velocity_scale)
        if self.velocity_scale <= 0:
            raise ValueError("velocity_scale must be > 0")

        self.mlp, self.velocity_mean_head, self.velocity_var_head = self._build_mlp_and_heads(
            input_dim=4,
            mlp_hidden_dims=mlp_hidden_dims,
            horizon=self.horizon,
        )

    def forward(
        self,
        current_velocity: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity_input = torch.as_tensor(current_velocity, dtype=torch.float32)
        velocity_device = velocity_input.device

        velocity_vec = self._prepare_vector_input(
            velocity_input,
            name="current_velocity",
            device=velocity_device,
        )
        goal_vec = self._prepare_vector_input(
            goal_position,
            name="goal_position",
            device=velocity_device,
            batch_size=int(velocity_vec.shape[0]),
        )

        velocity_vec = velocity_vec / self.velocity_scale
        mlp_input = torch.cat([velocity_vec, goal_vec], dim=1)
        return self._predict_from_features(mlp_input)


def build_proposal_network(
    input_shape: Sequence[int],
    horizon: int,
    channels: Sequence[int] = (32, 64, 128, 128, 128, 128),
    static_stem_channels: int = 8,
    downsample_strides: Sequence[int | Sequence[int]] = DEFAULT_PROPOSAL_DOWNSAMPLE_STRIDES,
    mlp_hidden_dims: Sequence[int] = (256, 256),
    velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
    min_variance: float = 1e-6,
    device: torch.device | str | None = None,
) -> VelocityTrajectoryProposalNetwork:
    model = VelocityTrajectoryProposalNetwork(
        input_shape=input_shape,
        horizon=horizon,
        channels=channels,
        static_stem_channels=static_stem_channels,
        downsample_strides=downsample_strides,
        mlp_hidden_dims=mlp_hidden_dims,
        velocity_scale=velocity_scale,
        min_variance=min_variance,
    )
    if device is not None:
        model = model.to(device)
    return model


def build_velocity_goal_mlp_proposal_network(
    horizon: int,
    mlp_hidden_dims: Sequence[int] = (128, 128),
    velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
    min_variance: float = 1e-6,
    device: torch.device | str | None = None,
) -> VelocityGoalMLPProposalNetwork:
    model = VelocityGoalMLPProposalNetwork(
        horizon=horizon,
        mlp_hidden_dims=mlp_hidden_dims,
        velocity_scale=velocity_scale,
        min_variance=min_variance,
    )
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "VelocityTrajectoryProposalNetwork",
    "VelocityGoalMLPProposalNetwork",
    "build_proposal_network",
    "build_velocity_goal_mlp_proposal_network",
]