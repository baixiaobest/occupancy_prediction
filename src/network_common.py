from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _check_size(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    b, c, t, h, w = x.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if h == out_h and w == out_w:
        return x

    raise ValueError(f"Expected input with spatial size ({out_h}, {out_w}), got ({h}, {w})")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _downsample_hw(hw: tuple[int, int], strides: Sequence[tuple[int, int]]) -> tuple[int, int]:
    h, w = int(hw[0]), int(hw[1])
    for sh, sw in strides:
        h = _ceil_div(h, sh)
        w = _ceil_div(w, sw)
    return h, w


class _DownsampleBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        # out = self.act(self.bn2(self.conv2(out)))
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
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        # out = self.act(self.bn2(self.conv2(out)))
        return out


__all__ = [
    "HUMAN_WALKING_SPEED_MPS",
    "_to_stride2",
    "_pack_video_time_to_channel",
    "_unpack_channel_to_video",
    "_check_size",
    "_downsample_hw",
    "_DownsampleBlock2d",
    "_UpsampleBlock2d",
]