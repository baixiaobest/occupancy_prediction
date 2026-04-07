from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DOWNSAMPLE_STRIDES: tuple[tuple[int, int], ...] = (
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
)

DEFAULT_UPSAMPLE_STRIDES: tuple[tuple[int, int], ...] = (
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
)

DEFAULT_UPSAMPLE_CHANNELS: tuple[int, ...] = (128, 64, 32, 16, 8, 4)
HUMAN_WALKING_SPEED_MPS: float = 1.4


def _to_stride2(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        stride = (int(value), int(value))
    else:
        if len(value) != 2:
            raise ValueError("stride must have 2 values: (h_stride, w_stride)")
        stride = (int(value[0]), int(value[1]))

    if any(s <= 0 for s in stride):
        raise ValueError("stride values must be > 0")
    return stride


def _deconv_params_from_stride2(stride: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    kernel = tuple(3 if s == 1 else 2 * s for s in stride)
    padding = tuple(1 if s == 1 else s // 2 for s in stride)
    output_padding = tuple(0 if s == 1 else s % 2 for s in stride)
    return kernel, padding, output_padding


def _pack_video_time_to_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError("Expected 5D tensor (B, C, T, H, W)")
    b, c, t, h, w = x.shape
    return x.reshape(b, c * t, h, w)


def _unpack_channel_to_video(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Expected 4D tensor (B, C, H, W)")
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")
    b, c, h, w = x.shape
    if c % time_steps != 0:
        raise ValueError(
            f"Channel dimension {c} is not divisible by time_steps={time_steps}"
        )
    c_per_t = c // time_steps
    return x.view(b, c_per_t, time_steps, h, w).contiguous()


def _resize_video_spatial(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    b, c, t, h, w = x.shape
    y = F.interpolate(
        x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    )
    _, c_out, h_out, w_out = y.shape
    return y.view(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()


class _DownsampleBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        return out


class _UpsampleBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        kernel_size, padding, output_padding = _deconv_params_from_stride2(stride)
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        return out


class VAEPredictionEncoder(nn.Module):
    """Fixed ResNet-style 3D CNN encoder for occupancy prediction VAE."""

    def __init__(
        self,
        input_shape: Sequence[int],
        latent_channel: int,
        channels: Sequence[int] = (32, 64, 128, 128, 128, 128),
        static_stem_channels: int = 8,
        velocity_mlp_dim: int = 16,
        velocity_condition_channels: int = 4,
        downsample_strides: Sequence[int | Sequence[int]] = DEFAULT_DOWNSAMPLE_STRIDES,
    ) -> None:
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError("input_shape must be (C, T, H, W)")

        in_channels = int(input_shape[0])
        # Kept for API compatibility. The latent representation is now a feature map
        # (B, c3, H/64, W/64), not a flattened vector of size `latent_dim`.
        self.latent_channel = int(latent_channel)
        if self.latent_channel <= 0:
            raise ValueError("latent_dim must be > 0")
        self.input_time_steps = int(input_shape[1])
        if self.input_time_steps <= 0:
            raise ValueError("input_shape time dimension must be > 0")
        self.static_stem_channels = int(static_stem_channels)
        if self.static_stem_channels <= 0:
            raise ValueError("static_stem_channels must be > 0")
        self.velocity_mlp_dim = int(velocity_mlp_dim)
        self.velocity_condition_channels = int(velocity_condition_channels)
        if self.velocity_mlp_dim <= 0:
            raise ValueError("velocity_mlp_dim must be > 0")
        if self.velocity_condition_channels < 0:
            raise ValueError("velocity_condition_channels must be >= 0")

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
        if self.velocity_condition_channels > 0:
            self.velocity_mlp = nn.Sequential(
                nn.Linear(2, self.velocity_mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.velocity_mlp_dim, self.velocity_condition_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.velocity_mlp = None
        down_blocks: list[nn.Module] = []
        in_ch = self.channels[0] + self.static_stem_channels + self.velocity_condition_channels
        for out_ch, stride in zip(self.channels[1:], stride_list):
            down_blocks.append(_DownsampleBlock2d(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)

        # Latent time is collapsed to 1 because temporal information is packed in channels.
        self.mu_head = nn.Conv2d(self.channels[-1], self.latent_channel, kernel_size=1)
        self.sigma_head = nn.Conv2d(self.channels[-1], self.latent_channel, kernel_size=1)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dynamic_x.ndim != 5:
            raise ValueError("dynamic_x must have shape (B, C, T, H, W)")
        if dynamic_x.shape[2] != self.input_time_steps:
            raise ValueError(
                f"dynamic_x time dimension must be {self.input_time_steps}, got {dynamic_x.shape[2]}"
            )

        if static_x.ndim == 4:
            # (B, 1, H, W) -> (B, 1, 1, H, W)
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

        h_dyn = self.dynamic_stem(_pack_video_time_to_channel(dynamic_x))
        h_static = self.static_stem(_pack_video_time_to_channel(static_x))
        if current_velocity is None:
            velocity_vec = torch.zeros((dynamic_x.shape[0], 2), dtype=torch.float32, device=dynamic_x.device)
        else:
            velocity_vec = torch.as_tensor(current_velocity, dtype=torch.float32, device=dynamic_x.device)
        velocity_vec = velocity_vec / float(HUMAN_WALKING_SPEED_MPS)
        velocity_embed = self.velocity_mlp(velocity_vec)
        velocity_map = velocity_embed.view(velocity_embed.shape[0], velocity_embed.shape[1], 1, 1)
        velocity_map = velocity_map.expand(-1, -1, h_dyn.shape[2], h_dyn.shape[3])
        h = torch.cat([h_dyn, h_static, velocity_map], dim=1)

        for down_block in self.down_blocks:
            h = down_block(h)

        mu = self.mu_head(h).unsqueeze(2)
        sigma = (F.softplus(self.sigma_head(h)) + 1e-6).unsqueeze(2)
        return mu, sigma

    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * eps."""
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class VAEPredictionDecoder(nn.Module):
    """Fixed ResNet-style 3D CNN decoder with convolutional context conditioning.

    Expects `z` as a latent feature map with shape `(B, C_latent, 1, H, W)`.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: Sequence[int],
        context_frames: int = 8,
        context_latent_channels: int | None = 32,
        downsample_channels: Sequence[int] = (8, 16, 32, 32, 32, 32),
        static_stem_channels: int = 8,
        velocity_mlp_dim: int = 16,
        velocity_condition_channels: int = 8,
        context_downsample_strides: Sequence[int | Sequence[int]] = DEFAULT_DOWNSAMPLE_STRIDES,
        upsample_channels: Sequence[int] | None = (
            128,
            64,
            32,
            16,
            8,
            4,
            2
        ),
        upsample_strides: Sequence[int | Sequence[int]] = DEFAULT_UPSAMPLE_STRIDES,
    ) -> None:
        super().__init__()

        if len(output_shape) != 4:
            raise ValueError("output_shape must be (C, T, H, W)")

        self.output_shape = tuple(int(v) for v in output_shape)
        self.context_frames = int(context_frames)
        if self.context_frames <= 0:
            raise ValueError("context_frames must be > 0")

        self.static_stem_channels = int(static_stem_channels)
        if self.static_stem_channels <= 0:
            raise ValueError("static_stem_channels must be > 0")
        self.velocity_mlp_dim = int(velocity_mlp_dim)
        self.velocity_condition_channels = int(velocity_condition_channels)
        if self.velocity_mlp_dim <= 0:
            raise ValueError("velocity_mlp_dim must be > 0")
        if self.velocity_condition_channels < 0:
            raise ValueError("velocity_condition_channels must be >= 0")

        latent_dim = int(latent_dim)

        self.latent_channels = latent_dim
        self.context_latent_channels = (
            self.latent_channels if context_latent_channels is None else int(context_latent_channels)
        )
        if self.context_latent_channels <= 0:
            raise ValueError("context_latent_channels must be > 0")

        self.downsample_channels = [int(c) for c in downsample_channels]
        if len(self.downsample_channels) < 2:
            raise ValueError("downsample_channels must contain at least 2 entries")
        if any(c <= 0 for c in self.downsample_channels):
            raise ValueError("all downsample_channels values must be > 0")

        if len(context_downsample_strides) != len(self.downsample_channels) - 1:
            raise ValueError("context_downsample_strides length must equal len(downsample_channels)-1")
        context_stride_list = [_to_stride2(s) for s in context_downsample_strides]

        if len(upsample_strides) == 0:
            raise ValueError("upsample_strides must contain at least 1 stride entry")
        upsample_stride_list = [_to_stride2(s) for s in upsample_strides]

        upsample_channels = [int(c) for c in upsample_channels]
        if len(upsample_channels) != len(upsample_stride_list) + 1:
            raise ValueError(
                "upsample_channels must have length len(upsample_strides)+1: "
                "[input_channels, block1_out, ..., blockN_out]"
            )

        self.context_dynamic_stem = nn.Sequential(
            nn.Conv2d(self.context_frames, self.downsample_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.downsample_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.context_static_stem = nn.Sequential(
            nn.Conv2d(self.context_frames, self.static_stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.static_stem_channels),
            nn.ReLU(inplace=True),
        )
        if self.velocity_condition_channels > 0:
            self.velocity_mlp = nn.Sequential(
                nn.Linear(2, self.velocity_mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.velocity_mlp_dim, self.velocity_condition_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.velocity_mlp = None
            self.velocity_to_map = None
        context_down_blocks: list[nn.Module] = []
        in_ch = self.downsample_channels[0] + self.static_stem_channels + self.velocity_condition_channels
        for out_ch, stride in zip(self.downsample_channels[1:], context_stride_list):
            context_down_blocks.append(_DownsampleBlock2d(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.context_down_blocks = nn.ModuleList(context_down_blocks)
        self.context_to_latent = nn.Conv2d(
            self.downsample_channels[-1],
            self.context_latent_channels,
            kernel_size=1,
        )

        merged_channels = self.latent_channels + self.context_latent_channels
        self.input_proj = nn.Conv2d(merged_channels, upsample_channels[0], kernel_size=1)

        self.upsample_blocks = nn.ModuleList(
            [
                _UpsampleBlock2d(
                    upsample_channels[i],
                    upsample_channels[i + 1],
                    stride=s,
                )
                for i, s in enumerate(upsample_stride_list)
            ]
        )
        self.to_output = nn.Conv2d(
            upsample_channels[-1],
            self.output_shape[0] * self.output_shape[1],
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        z: torch.Tensor,
        dynamic_context: torch.Tensor,
        static_x: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError("z must have shape (B, C_latent, 1, H, W)")
        if z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected z channel dimension {self.latent_channels}, got {z.shape[1]}"
            )
        if z.shape[2] != 1:
            raise ValueError(
                f"Expected z time dimension 1, got {z.shape[2]}"
            )

        if dynamic_context.ndim != 5:
            raise ValueError("dynamic_context must have shape (B, 1, T_ctx, H, W)")
        if dynamic_context.shape[1] != 1:
            raise ValueError("dynamic_context channel dimension must be 1")
        if dynamic_context.shape[2] != self.context_frames:
            raise ValueError(
                f"Expected dynamic_context time dimension {self.context_frames}, got {dynamic_context.shape[2]}"
            )

        if static_x.ndim == 4:
            static_x = static_x.unsqueeze(2)
        if static_x.ndim != 5:
            raise ValueError("static_x must have shape (B, 1, H, W) or (B, 1, T, H, W)")
        if static_x.shape[1] != 1:
            raise ValueError("static_x channel dimension must be 1")
        if static_x.shape[-2:] != dynamic_context.shape[-2:]:
            raise ValueError("static_x and dynamic_context must share H/W")
        if static_x.shape[2] == 1 and dynamic_context.shape[2] > 1:
            static_x = static_x.expand(-1, -1, dynamic_context.shape[2], -1, -1)
        elif static_x.shape[2] != dynamic_context.shape[2]:
            raise ValueError("static_x time dimension must be 1 or match dynamic_context")

        cond_dyn = self.context_dynamic_stem(_pack_video_time_to_channel(dynamic_context))
        cond_static = self.context_static_stem(_pack_video_time_to_channel(static_x))
        if current_velocity is None:
            velocity_vec = torch.zeros((dynamic_context.shape[0], 2), dtype=torch.float32, device=dynamic_context.device)
        else:
            velocity_vec = torch.as_tensor(current_velocity, dtype=torch.float32, device=dynamic_context.device)
        velocity_vec = velocity_vec / float(HUMAN_WALKING_SPEED_MPS)
        velocity_embed = self.velocity_mlp(velocity_vec)
        velocity_map = velocity_embed.view(velocity_embed.shape[0], velocity_embed.shape[1], 1, 1)
        velocity_map = velocity_map.expand(-1, -1, cond_dyn.shape[2], cond_dyn.shape[3])
        cond = torch.cat([cond_dyn, cond_static, velocity_map], dim=1)
        for down_block in self.context_down_blocks:
            cond = down_block(cond)
        cond = self.context_to_latent(cond).unsqueeze(2)
        cond = _resize_video_spatial(cond, (z.shape[3], z.shape[4]))

        merged = torch.cat([z, cond], dim=1)

        h = self.input_proj(_pack_video_time_to_channel(merged))
        for up_block in self.upsample_blocks:
            h = up_block(h)
        h = _unpack_channel_to_video(self.to_output(h), self.output_shape[1])

        h = _resize_video_spatial(h, (self.output_shape[2], self.output_shape[3]))
        return h


def build_prediction_vae_models(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    latent_channel: int,
    channels: Sequence[int] = (32, 64, 128, 128, 128, 128),
    decoder_downsample_channels: Sequence[int] | None = None,
    decoder_context_latent_channel: int | None = None,
    static_stem_channels: int = 8,
    velocity_mlp_dim: int = 16,
    encoder_velocity_condition_channels: int = 8,
    decoder_velocity_condition_channels: int = 8,
    decoder_context_frames: int = 8,
    downsample_strides: Sequence[int | Sequence[int]] = DEFAULT_DOWNSAMPLE_STRIDES,
    decoder_context_downsample_strides: Sequence[int | Sequence[int]] | None = None,
    upsample_strides: Sequence[int | Sequence[int]] = DEFAULT_UPSAMPLE_STRIDES,
    upsample_channels: Sequence[int] = DEFAULT_UPSAMPLE_CHANNELS,
    device: torch.device | str | None = None,
) -> tuple[VAEPredictionEncoder, VAEPredictionDecoder]:
    """Build a consistent encoder/decoder pair for occupancy prediction."""
    if any(int(c) <= 0 for c in channels):
        raise ValueError("channels must contain positive integers")

    context_downsample_strides = (
        downsample_strides if decoder_context_downsample_strides is None else decoder_context_downsample_strides
    )
    context_downsample_strides = [_to_stride2(s) for s in context_downsample_strides]

    decoder_downsample_channels = [int(c) for c in decoder_downsample_channels]
    if len(decoder_downsample_channels) != len(context_downsample_strides) + 1:
        raise ValueError("decoder_downsample_channels length must equal len(context_downsample_strides)+1")

    encoder = VAEPredictionEncoder(
        input_shape=input_shape,
        latent_channel=latent_channel,
        channels=channels,
        static_stem_channels=static_stem_channels,
        velocity_mlp_dim=velocity_mlp_dim,
        velocity_condition_channels=encoder_velocity_condition_channels,
        downsample_strides=downsample_strides,
    )
    decoder = VAEPredictionDecoder(
        latent_dim=latent_channel,
        output_shape=output_shape,
        context_frames=decoder_context_frames,
        context_latent_channels=decoder_context_latent_channel,
        downsample_channels=decoder_downsample_channels,
        static_stem_channels=static_stem_channels,
        velocity_mlp_dim=velocity_mlp_dim,
        velocity_condition_channels=decoder_velocity_condition_channels,
        context_downsample_strides=context_downsample_strides,
        upsample_channels=upsample_channels,
        upsample_strides=upsample_strides,
    )

    if device is not None:
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    return encoder, decoder
