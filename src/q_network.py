from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .network_common import HUMAN_WALKING_SPEED_MPS


def _build_mlp(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Module, int]:
    dims = [int(d) for d in hidden_dims]
    if any(d <= 0 for d in dims):
        raise ValueError("MLP hidden dimensions must be > 0")

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(dim))
        last_dim = dim

    if len(layers) == 0:
        return nn.Identity(), int(input_dim)
    return nn.Sequential(*layers), last_dim


class VelocityPlanQNetwork(nn.Module):
    """Action-conditioned Q-network for velocity-plan evaluation.

    Inputs:
    - goal_position: (B, 2)
    - current_velocity: (B, 2)
    - planned_velocities: (B, horizon, 2)
    - tapped_future_features:
        * (B, horizon, C, H, W), or
        * (B, horizon * C, H, W)

    Output:
    - q_values: (B,)

    Notes:
    - `tapped_future_features` should come from decoder taps over the full horizon.
    - `planned_velocities` is the candidate action sequence being scored by Q.
    """

    def __init__(
        self,
        tapped_feature_channels: int,
        horizon: int,
        spatial_channels: Sequence[int] = (128, 128, 128),
        spatial_strides: Sequence[int] = (1, 2, 2),
        plan_conv_channels: Sequence[int] = (32, 64, 64),
        plan_kernel_size: int = 3,
        state_mlp_dims: Sequence[int] = (32, 32),
        fusion_mlp_dims: Sequence[int] = (128, 64),
        velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
        goal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        """
        Constructor args summary:
        tapped_feature_channels: Conv2d input channels for packed taps, usually horizon * C_tap.
        horizon: Number of future steps in planned_velocities and tapped features.
        spatial_channels: Conv2d output channels per spatial layer; last entry is spatial embedding dim.
        spatial_strides: Conv2d stride per spatial layer; same length as spatial_channels.
        plan_conv_channels: Conv1d output channels per plan layer; last entry is plan embedding dim.
        plan_kernel_size: Conv1d kernel size for planned-velocity branch (odd values recommended).
        state_mlp_dims: Hidden widths for MLP over [goal_position, current_velocity].
        fusion_mlp_dims: Hidden widths for fused [spatial, plan, state] embedding.
        velocity_scale: Normalization divisor for current/planned velocity values.
        goal_scale: Normalization divisor for goal_position coordinates.
        """

        # Packed spatial channels expected by the Conv2d branch.
        # If each future tap has C channels and there are `horizon` steps,
        # then this value is typically horizon * C.
        self.tapped_feature_channels = int(tapped_feature_channels)
        # Number of planned future velocity steps and tapped future feature steps.
        self.horizon = int(horizon)
        self.velocity_scale = float(velocity_scale)
        self.goal_scale = float(goal_scale)

        if self.tapped_feature_channels <= 0:
            raise ValueError("tapped_feature_channels must be > 0")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.velocity_scale <= 0:
            raise ValueError("velocity_scale must be > 0")
        if self.goal_scale <= 0:
            raise ValueError("goal_scale must be > 0")

        spatial_channels_list = [int(c) for c in spatial_channels]
        spatial_strides_list = [int(s) for s in spatial_strides]
        if len(spatial_channels_list) == 0:
            raise ValueError("spatial_channels must contain at least one value")
        if len(spatial_channels_list) != len(spatial_strides_list):
            raise ValueError("spatial_channels and spatial_strides must have the same length")
        if any(c <= 0 for c in spatial_channels_list):
            raise ValueError("spatial_channels values must be > 0")
        if any(s <= 0 for s in spatial_strides_list):
            raise ValueError("spatial_strides values must be > 0")

        spatial_layers: list[nn.Module] = []
        in_spatial = self.tapped_feature_channels
        for out_spatial, stride in zip(spatial_channels_list, spatial_strides_list):
            spatial_layers.append(
                nn.Conv2d(in_spatial, out_spatial, kernel_size=3, stride=stride, padding=1, bias=False)
            )
            spatial_layers.append(nn.BatchNorm2d(out_spatial))
            spatial_layers.append(nn.ReLU(inplace=True))
            in_spatial = out_spatial
        self.spatial_conv = nn.Sequential(*spatial_layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_out_dim = spatial_channels_list[-1]

        plan_channels_list = [int(c) for c in plan_conv_channels]
        if len(plan_channels_list) == 0:
            raise ValueError("plan_conv_channels must contain at least one value")
        if any(c <= 0 for c in plan_channels_list):
            raise ValueError("plan_conv_channels values must be > 0")
        plan_kernel_size = int(plan_kernel_size)
        if plan_kernel_size <= 0:
            raise ValueError("plan_kernel_size must be > 0")
        plan_padding = plan_kernel_size // 2

        plan_layers: list[nn.Module] = []
        in_plan = 2
        for out_plan in plan_channels_list:
            plan_layers.append(
                nn.Conv1d(in_plan, out_plan, kernel_size=plan_kernel_size, stride=1, padding=plan_padding, bias=False)
            )
            plan_layers.append(nn.BatchNorm1d(out_plan))
            plan_layers.append(nn.ReLU(inplace=True))
            in_plan = out_plan
        self.plan_conv = nn.Sequential(*plan_layers)
        self.plan_pool = nn.AdaptiveAvgPool1d(1)
        self.plan_out_dim = plan_channels_list[-1]

        self.state_mlp, self.state_out_dim = _build_mlp(input_dim=4, hidden_dims=state_mlp_dims)

        fusion_in_dim = self.spatial_out_dim + self.plan_out_dim + self.state_out_dim
        self.fusion_mlp, fusion_out_dim = _build_mlp(input_dim=fusion_in_dim, hidden_dims=fusion_mlp_dims)
        self.q_head = nn.Linear(fusion_out_dim, 1)

    @staticmethod
    def _prepare_vector_input(
        value: torch.Tensor,
        *,
        name: str,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        vec = torch.as_tensor(value, dtype=torch.float32, device=device)
        if vec.ndim == 1:
            if vec.shape[0] != 2:
                raise ValueError(f"{name} 1D input must have shape (2,)")
            vec = vec.unsqueeze(0)
        if vec.ndim != 2 or vec.shape[1] != 2:
            raise ValueError(f"{name} must have shape (B, 2)")
        if vec.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must be {batch_size}, got {vec.shape[0]}")
        return vec

    def _prepare_tapped_features(self, tapped_future_features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(tapped_future_features, dtype=torch.float32)
        if features.ndim == 5:
            batch_size, horizon, channels, height, width = features.shape
            if horizon != self.horizon:
                raise ValueError(f"tapped feature horizon must be {self.horizon}, got {horizon}")
            packed = features.reshape(batch_size, horizon * channels, height, width)
        elif features.ndim == 4:
            packed = features
        else:
            raise ValueError(
                "tapped_future_features must have shape (B, horizon, C, H, W) "
                "or (B, horizon*C, H, W)"
            )

        if packed.shape[1] != self.tapped_feature_channels:
            raise ValueError(
                "Packed tapped feature channels mismatch: "
                f"expected {self.tapped_feature_channels}, got {packed.shape[1]}"
            )
        return packed

    def forward(
        self,
        goal_position: torch.Tensor,
        current_velocity: torch.Tensor,
        planned_velocities: torch.Tensor,
        tapped_future_features: torch.Tensor,
    ) -> torch.Tensor:
        # goal_position: (B, 2)
        # current_velocity: (B, 2)
        # planned_velocities: (B, horizon, 2)
        # tapped_future_features: (B, horizon, C, H, W) or packed (B, horizon*C, H, W)
        plan = torch.as_tensor(planned_velocities, dtype=torch.float32)
        if plan.ndim != 3 or plan.shape[2] != 2:
            raise ValueError("planned_velocities must have shape (B, horizon, 2)")
        if plan.shape[1] != self.horizon:
            raise ValueError(f"planned_velocities horizon must be {self.horizon}, got {plan.shape[1]}")

        batch_size = int(plan.shape[0])
        device = plan.device

        goal_vec = self._prepare_vector_input(
            goal_position,
            name="goal_position",
            device=device,
            batch_size=batch_size,
        ) / self.goal_scale
        current_velocity_vec = self._prepare_vector_input(
            current_velocity,
            name="current_velocity",
            device=device,
            batch_size=batch_size,
        ) / self.velocity_scale

        packed_tapped = self._prepare_tapped_features(tapped_future_features).to(device=device)
        if packed_tapped.shape[0] != batch_size:
            raise ValueError(
                "tapped_future_features batch size mismatch: "
                f"expected {batch_size}, got {packed_tapped.shape[0]}"
            )

        spatial_feat = self.spatial_conv(packed_tapped)
        spatial_feat = self.spatial_pool(spatial_feat).flatten(start_dim=1)

        plan_feat = plan.transpose(1, 2).contiguous()
        plan_feat = self.plan_conv(plan_feat)
        plan_feat = self.plan_pool(plan_feat).flatten(start_dim=1)

        state_input = torch.cat([goal_vec, current_velocity_vec], dim=1)
        state_feat = self.state_mlp(state_input)

        fused = torch.cat([spatial_feat, plan_feat, state_feat], dim=1)
        fused = self.fusion_mlp(fused)
        q_values = self.q_head(fused).squeeze(-1)
        return q_values


def build_q_network(
    tapped_feature_channels: int,
    horizon: int,
    spatial_channels: Sequence[int] = (256, 256, 128),
    spatial_strides: Sequence[int] = (1, 2, 2),
    plan_conv_channels: Sequence[int] = (32, 64, 64),
    plan_kernel_size: int = 3,
    state_mlp_dims: Sequence[int] = (64, 64),
    fusion_mlp_dims: Sequence[int] = (256, 128),
    velocity_scale: float = HUMAN_WALKING_SPEED_MPS,
    goal_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> VelocityPlanQNetwork:
    """Build a VelocityPlanQNetwork with the same argument semantics as the class.

    Args:
    - tapped_feature_channels:
        Packed tapped-feature channels expected by Conv2d input (typically horizon * C_tap).
    - horizon:
        Planned-action/tapped-feature horizon length.
    - spatial_channels:
        Conv2d output channels for the tapped-feature branch.
    - spatial_strides:
        Conv2d stride for each spatial branch layer.
    - plan_conv_channels:
        Conv1d output channels for the planned-velocity branch.
    - plan_kernel_size:
        Conv1d kernel size for the planned-velocity branch.
    - state_mlp_dims:
        MLP hidden widths for `[goal_position, current_velocity]`.
    - fusion_mlp_dims:
        MLP hidden widths after branch fusion.
    - velocity_scale:
        Velocity normalization scale.
    - goal_scale:
        Goal-position normalization scale.
    - device:
        Optional device to place the model on after construction.
    """
    model = VelocityPlanQNetwork(
        tapped_feature_channels=tapped_feature_channels,
        horizon=horizon,
        spatial_channels=spatial_channels,
        spatial_strides=spatial_strides,
        plan_conv_channels=plan_conv_channels,
        plan_kernel_size=plan_kernel_size,
        state_mlp_dims=state_mlp_dims,
        fusion_mlp_dims=fusion_mlp_dims,
        velocity_scale=velocity_scale,
        goal_scale=goal_scale,
    )
    if device is not None:
        model = model.to(device)
    return model


__all__ = ["VelocityPlanQNetwork", "build_q_network"]
