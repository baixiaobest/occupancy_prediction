from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_stride3(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        stride = (int(value), int(value), int(value))
    else:
        if len(value) != 3:
            raise ValueError("stride must have 3 values: (t_stride, h_stride, w_stride)")
        stride = (int(value[0]), int(value[1]), int(value[2]))

    if any(s <= 0 for s in stride):
        raise ValueError("stride values must be > 0")
    return stride


def _deconv_params_from_stride(stride: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    # For stride=1: use (k=3, p=1, op=0) to preserve size.
    # For stride>1: use (k=2s, p=s//2, op=s%2) to approximately scale by stride.
    kernel = tuple(3 if s == 1 else 2 * s for s in stride)
    padding = tuple(1 if s == 1 else s // 2 for s in stride)
    output_padding = tuple(0 if s == 1 else s % 2 for s in stride)
    return kernel, padding, output_padding


class _ResidualBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class _DownsampleResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | Sequence[int] = 2,
    ) -> None:
        super().__init__()
        stride_3d = _to_stride3(stride)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride_3d,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride_3d,
            bias=False,
        )
        self.skip_bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_bn(self.skip(x))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class _UpsampleResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | Sequence[int] = 2,
    ) -> None:
        super().__init__()
        stride_3d = _to_stride3(stride)
        kernel_size, padding, output_padding = _deconv_params_from_stride(stride_3d)
        self.conv1 = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride_3d,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride_3d,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.skip_bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_bn(self.skip(x))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class _UpsampleInterpResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | Sequence[int] = 2,
        mode: str = "trilinear",
    ) -> None:
        super().__init__()
        self.scale_factor = _to_stride3(stride)
        self.mode = str(mode).lower()
        if self.mode not in {"nearest", "trilinear"}:
            raise ValueError("mode must be either 'nearest' or 'trilinear'")

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "trilinear":
            x_up = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        else:
            x_up = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        residual = self.skip_bn(self.skip(x_up))
        out = self.act(self.bn1(self.conv1(x_up)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class VAEPredictionEncoder(nn.Module):
    """Fixed ResNet-style 3D CNN encoder for occupancy prediction VAE."""

    def __init__(
        self,
        input_shape: Sequence[int],
        latent_channel: int,
        base_channels: int = 32,
        downsample_strides: Sequence[int | Sequence[int]] = (
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
    ) -> None:
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError("input_shape must be (C, T, H, W)")

        in_channels = int(input_shape[0])
        # Kept for API compatibility. The latent representation is now a feature map
        # (B, c3, T/4, H/64, W/64), not a flattened vector of size `latent_dim`.
        self.latent_channel = int(latent_channel)
        if self.latent_channel <= 0:
            raise ValueError("latent_dim must be > 0")

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c2 * 2

        if len(downsample_strides) < 2:
            raise ValueError("downsample_strides must contain at least 2 stride entries")

        stride_list = [_to_stride3(s) for s in downsample_strides]
        s1, s2 = stride_list[0], stride_list[1]

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        # Input `x` shape: (B, C, T, H, W)
        # After `stem`: (B, c1, T, H, W)
        self.res1 = _ResidualBlock3d(c1)
        # After `res1`: (B, c1, T, H, W)
        self.down1 = _DownsampleResBlock3d(c1, c2, stride=s1)
        # After `down1`: (B, c2, T/s1_t, H/s1_h, W/s1_w)
        self.res2 = _ResidualBlock3d(c2)
        # After `res2`: (B, c2, T/s1_t, H/s1_h, W/s1_w)
        self.down2 = _DownsampleResBlock3d(c2, c3, stride=s2)
        # After `down2`: (B, c3, T/(s1_t*s2_t), H/(s1_h*s2_h), W/(s1_w*s2_w))
        self.res3 = _ResidualBlock3d(c3)
        # After `res3`: same as after `down2`
        self.extra_spatial_down = nn.ModuleList(
            [_DownsampleResBlock3d(c3, c3, stride=s) for s in stride_list[2:]]
        )
        # With default `downsample_strides`, shape progression is:
        # (2,2,2), (2,2,2), then four (1,2,2)
        # -> (B, c3, T/4, H/64, W/64)

        # Mu is map via 1x1x1 conv.
        # Sigma is a same-shape positive map via 1x1x1 conv + softplus.
        self.mu_head = nn.Conv3d(c3, self.latent_channel, kernel_size=1)
        self.sigma_head = nn.Conv3d(c3, self.latent_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.res1(h)    # (B, c1, T, H, W)
        h = self.down1(h)   # (B, c2, T/s1_t, H/s1_h, W/s1_w)
        h = self.res2(h)    # (B, c2, T/s1_t, H/s1_h, W/s1_w)
        h = self.down2(h)   # (B, c3, T/(s1_t*s2_t), H/(s1_h*s2_h), W/(s1_w*s2_w))
        h = self.res3(h)    # (B, c3, T/(s1_t*s2_t), H/(s1_h*s2_h), W/(s1_w*s2_w))
        for down_block in self.extra_spatial_down:
            h = down_block(h)  # Additional blocks follow remaining entries in `downsample_strides`.

        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        return mu, sigma

    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * eps."""
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class VAEPredictionDecoder(nn.Module):
    """Fixed ResNet-style 3D CNN decoder with optional conditioning.

    Expects `z` as a latent feature map with shape
    `(B, C3, T/s_t, H/s_h, W/s_w)` from the encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: Sequence[int],
        conditional_dim: int | None = None,
        condition_embed_dim: int = 16,
        condition_mlp_hidden_dim: int = 64,
        upsample_channels: Sequence[int] | None = (
            128,
            64,
            32,
            16,
            8,
            4,
            2
        ),
        upsample_strides: Sequence[int | Sequence[int]] = (
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        upsample_method: str = "transpose",
        upsample_interp_mode: str = "trilinear",
    ) -> None:
        super().__init__()

        if len(output_shape) != 4:
            raise ValueError("output_shape must be (C, T, H, W)")
        if int(output_shape[1]) != 1:
            raise ValueError("Decoder output must be one-step prediction, so output_shape[1] must be 1")

        self.output_shape = tuple(int(v) for v in output_shape)
        self.conditional_dim = int(conditional_dim) if conditional_dim is not None else None
        self.condition_embed_dim = int(condition_embed_dim)
        self.condition_mlp_hidden_dim = int(condition_mlp_hidden_dim)

        latent_dim = int(latent_dim)

        self.latent_channels = latent_dim

        if len(upsample_strides) == 0:
            raise ValueError("upsample_strides must contain at least 1 stride entry")
        upsample_stride_list = [_to_stride3(s) for s in upsample_strides]

        self.upsample_method = str(upsample_method).lower()
        if self.upsample_method not in {"transpose", "interpolate"}:
            raise ValueError("upsample_method must be either 'transpose' or 'interpolate'")
        self.upsample_interp_mode = str(upsample_interp_mode).lower()
        if self.upsample_interp_mode not in {"nearest", "trilinear"}:
            raise ValueError("upsample_interp_mode must be either 'nearest' or 'trilinear'")

        upsample_channels = [int(c) for c in upsample_channels]
        if len(upsample_channels) != len(upsample_stride_list) + 1:
            raise ValueError(
                "upsample_channels must have length len(upsample_strides)+1: "
                "[input_channels, block1_out, ..., blockN_out]"
            )

        if self.conditional_dim is not None:
            self.condition_mlp = nn.Sequential(
                nn.Linear(self.conditional_dim, self.condition_mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.condition_mlp_hidden_dim, self.condition_embed_dim),
            )
            decoder_in_channels = self.latent_channels + self.condition_embed_dim
        else:
            self.condition_mlp = None
            decoder_in_channels = self.latent_channels

        self.res_in = _ResidualBlock3d(decoder_in_channels)
        self.input_proj = nn.Conv3d(decoder_in_channels, upsample_channels[0], kernel_size=1)

        upsample_block_cls = (
            _UpsampleInterpResBlock3d if self.upsample_method == "interpolate" else _UpsampleResBlock3d
        )
        if self.upsample_method == "interpolate":
            self.upsample_blocks = nn.ModuleList(
                [
                    upsample_block_cls(
                        upsample_channels[i],
                        upsample_channels[i + 1],
                        stride=s,
                        mode=self.upsample_interp_mode,
                    )
                    for i, s in enumerate(upsample_stride_list)
                ]
            )
        else:
            self.upsample_blocks = nn.ModuleList(
                [
                    upsample_block_cls(upsample_channels[i], upsample_channels[i + 1], stride=s)
                    for i, s in enumerate(upsample_stride_list)
                ]
            )
        self.res_out = _ResidualBlock3d(upsample_channels[-1])

        self.to_output = nn.Conv3d(upsample_channels[-1], self.output_shape[0], kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError("z must have shape (B, C3, T/s_t, H/s_h, W/s_w)")
        if z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected z channel dimension {self.latent_channels}, got {z.shape[1]}"
            )

        if self.conditional_dim is not None:
            if condition.ndim != 2:
                condition = torch.flatten(condition, start_dim=1)
            if condition.shape[-1] != self.conditional_dim:
                raise ValueError(
                    f"Expected condition last dimension {self.conditional_dim}, got {condition.shape[-1]}"
                )
            cond_embed = self.condition_mlp(condition)
            cond_map = cond_embed.view(cond_embed.shape[0], cond_embed.shape[1], 1, 1, 1)
            cond_map = cond_map.expand(-1, -1, z.shape[2], z.shape[3], z.shape[4])
            z = torch.cat([z, cond_map], dim=1)

        h = self.res_in(z)
        h = self.input_proj(h)
        for up_block in self.upsample_blocks:
            h = up_block(h)
        h = self.res_out(h)
        h = self.to_output(h)

        # Match the requested temporal-spatial size even when strides do not align exactly.
        # Final output is one-step prediction: (B, C_out, 1, H, W)
        h = F.interpolate(h, size=self.output_shape[1:], mode="trilinear", align_corners=False)
        return h
