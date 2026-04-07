from __future__ import annotations

import math

import torch


def slice_centered_patch(
    grid: torch.Tensor,
    center_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution_xy: tuple[float, float],
    patch_shape: tuple[int, int],
    *,
    binary: bool,
    prefer_view: bool,
) -> torch.Tensor:
    """Slice a local centered patch with optional binarization and view fast-path."""
    grid_2d = torch.as_tensor(grid, dtype=torch.float32)
    if grid_2d.ndim == 3 and grid_2d.shape[0] == 1:
        grid_2d = grid_2d[0]
    if grid_2d.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {tuple(grid_2d.shape)}")

    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    res_x, res_y = float(resolution_xy[0]), float(resolution_xy[1])
    center_x = float(center_xy[0].item())
    center_y = float(center_xy[1].item())
    half_w = 0.5 * patch_w * res_x
    half_h = 0.5 * patch_h * res_y

    start_x = int(math.floor((center_x - half_w - origin_xy[0]) / res_x))
    start_y = int(math.floor((center_y - half_h - origin_xy[1]) / res_y))
    end_x = start_x + patch_w
    end_y = start_y + patch_h

    in_bounds = (
        start_x >= 0
        and start_y >= 0
        and end_x <= int(grid_2d.shape[1])
        and end_y <= int(grid_2d.shape[0])
    )
    if prefer_view and in_bounds:
        patch = grid_2d[start_y:end_y, start_x:end_x]
        return (patch > 0).float() if binary else patch

    out = torch.zeros((patch_h, patch_w), dtype=torch.float32)
    src_x0 = max(0, start_x)
    src_y0 = max(0, start_y)
    src_x1 = min(int(grid_2d.shape[1]), end_x)
    src_y1 = min(int(grid_2d.shape[0]), end_y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - start_x
    dst_y0 = src_y0 - start_y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = grid_2d[src_y0:src_y1, src_x0:src_x1]
    return (out > 0).float() if binary else out
