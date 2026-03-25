from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.rollout_data import RollOutData, SceneRollOutData


@dataclass
class DatasetStats:
    num_scene_files: int
    num_train_agent_sequences: int
    num_val_agent_sequences: int
    num_train_anchors: int
    num_val_anchors: int
    num_train_samples: int
    num_val_samples: int


class OccupancyWindowDataset(Dataset):
    """Anchor-window dataset over precomputed local occupancy windows."""

    def __init__(
        self,
        sequences: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        history_len: int = 16,
        future_len: int = 8,
        decoder_context_len: int = 8,
        stride: int = 1,
    ) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be > 0")
        if future_len <= 0:
            raise ValueError("future_len must be > 0")
        if decoder_context_len <= 0:
            raise ValueError("decoder_context_len must be > 0")
        if decoder_context_len > history_len:
            raise ValueError("decoder_context_len must be <= history_len")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.decoder_context_len = int(decoder_context_len)
        self.window_size = self.history_len + self.future_len
        self.samples = list(sequences)

        for seq, static_map, velocity in self.samples:
            if seq.ndim != 3:
                raise ValueError("Each sequence must have shape (history+future, H, W)")
            if seq.shape[0] != self.window_size:
                raise ValueError("Each sequence must have exactly history_len + future_len frames")
            if static_map.ndim != 2:
                raise ValueError("Each static map must have shape (H, W)")
            if static_map.shape != seq.shape[-2:]:
                raise ValueError("Static map spatial shape must match sequence shape")
            vel = torch.as_tensor(velocity, dtype=torch.float32)
            if vel.shape != (2,):
                raise ValueError("Each velocity must have shape (2,)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, current_static, current_velocity = self.samples[index]

        past = seq[: self.history_len]
        future = seq[self.history_len : self.window_size]

        x_encoder_dynamic = torch.cat([past, future], dim=0).unsqueeze(0)
        x_decoder_dynamic = past[-self.decoder_context_len :].unsqueeze(0)
        x_static = current_static.unsqueeze(0)
        y = future.unsqueeze(0)
        return x_encoder_dynamic, x_decoder_dynamic, x_static, current_velocity, y


def _load_agent_sequences_from_file(
    pt_path: Path,
    history_len: int,
    future_len: int,
    anchor_stride: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    try:
        payload = torch.load(pt_path, map_location="cpu")
    except AttributeError as exc:
        # Legacy pickle payloads reference removed classes (e.g., AnchorRollOutData).
        raise ValueError(
            f"Failed to load {pt_path}: legacy rollout format is no longer supported. "
            "Regenerate rollout .pt files using the current scripts/ORCA_rollout.py format."
        ) from exc

    def _to_2d(frame: object) -> torch.Tensor:
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
        return (tensor > 0).float()

    def _slice_centered_patch(
        grid: torch.Tensor,
        center_xy: torch.Tensor,
        origin_xy: tuple[float, float],
        resolution_xy: tuple[float, float],
        patch_shape: tuple[int, int],
    ) -> torch.Tensor:
        patch_h, patch_w = patch_shape
        res_x, res_y = float(resolution_xy[0]), float(resolution_xy[1])
        center_x = float(center_xy[0].item())
        center_y = float(center_xy[1].item())
        half_w = 0.5 * patch_w * res_x
        half_h = 0.5 * patch_h * res_y

        start_x = int(np.floor((center_x - half_w - origin_xy[0]) / res_x))
        start_y = int(np.floor((center_y - half_h - origin_xy[1]) / res_y))

        out = torch.zeros((patch_h, patch_w), dtype=torch.float32)
        src_x0 = max(0, start_x)
        src_y0 = max(0, start_y)
        src_x1 = min(grid.shape[1], start_x + patch_w)
        src_y1 = min(grid.shape[0], start_y + patch_h)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out

        dst_x0 = src_x0 - start_x
        dst_y0 = src_y0 - start_y
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = grid[src_y0:src_y1, src_x0:src_x1]
        return (out > 0).float()

    def _from_scene_obj(scene: SceneRollOutData) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if scene.scene_static_map is None:
            raise ValueError("RollOutData.scene_static_map is required")
        if scene.scene_map_origin is None:
            raise ValueError("RollOutData.scene_map_origin is required")
        if scene.local_map_shape is None:
            raise ValueError("RollOutData.local_map_shape is required")
        if scene.scene_dynamic_maps is None:
            raise ValueError("RollOutData.scene_dynamic_maps is required")
        out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        scene_static = _to_2d(scene.scene_static_map)
        scene_origin = (
            float(scene.scene_map_origin[0]),
            float(scene.scene_map_origin[1]),
        )
        resolution_xy = (
            float(scene.occupancy_resolution[0]),
            float(scene.occupancy_resolution[1]),
        )
        patch_shape = (int(scene.local_map_shape[0]), int(scene.local_map_shape[1]))
        dynamic_maps = torch.as_tensor(scene.scene_dynamic_maps, dtype=torch.float32)
        if dynamic_maps.ndim != 4:
            raise ValueError("scene_dynamic_maps must have shape (num_agents, total_time, H, W)")

        window_size = int(history_len + future_len)

        for agent_idx in sorted(scene.agents.keys()):
            agent_data = scene.agents[agent_idx]
            if agent_idx < 0 or agent_idx >= dynamic_maps.shape[0]:
                continue

            anchor_times = agent_data.anchor_times
            anchor_centers = agent_data.anchor_centers
            velocity_series = agent_data.current_velocities
            if not anchor_times:
                continue
            if anchor_centers is None or velocity_series is None:
                raise ValueError("Agent metadata requires anchor_centers and current_velocities")

            centers_tensor = torch.as_tensor(anchor_centers, dtype=torch.float32)
            velocity_tensor = torch.as_tensor(velocity_series, dtype=torch.float32)
            if centers_tensor.shape != (len(anchor_times), 2):
                raise ValueError("anchor_centers must have shape (num_anchors, 2)")
            if velocity_tensor.shape != (len(anchor_times), 2):
                raise ValueError("current_velocities must have shape (num_anchors, 2)")
            agent_dynamic = dynamic_maps[agent_idx]
            total_time = int(agent_dynamic.shape[0])

            anchor_indices = list(range(0, len(anchor_times), max(1, int(anchor_stride))))
            for anchor_meta_idx in anchor_indices:
                anchor_t = int(anchor_times[anchor_meta_idx])
                start_t = anchor_t - int(history_len)
                end_t = anchor_t + int(future_len)
                if start_t < 0 or end_t > total_time:
                    continue

                center_xy = centers_tensor[anchor_meta_idx]
                static_local = _slice_centered_patch(
                    grid=scene_static,
                    center_xy=center_xy,
                    origin_xy=scene_origin,
                    resolution_xy=resolution_xy,
                    patch_shape=patch_shape,
                )

                dynamic_window: list[torch.Tensor] = []
                for absolute_t in range(start_t, end_t):
                    dynamic_global_t = _to_2d(agent_dynamic[absolute_t])
                    dynamic_local_t = _slice_centered_patch(
                        grid=dynamic_global_t,
                        center_xy=center_xy,
                        origin_xy=scene_origin,
                        resolution_xy=resolution_xy,
                        patch_shape=patch_shape,
                    )
                    dynamic_window.append(dynamic_local_t)

                if len(dynamic_window) != window_size:
                    continue

                out.append(
                    (
                        torch.stack(dynamic_window, dim=0),
                        static_local,
                        velocity_tensor[anchor_meta_idx].to(dtype=torch.float32),
                    )
                )

        return out

    sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if isinstance(payload, RollOutData):
        for scene in payload.scenes:
            sequences.extend(_from_scene_obj(scene))
    else:
        raise ValueError(f"Unsupported payload format in {pt_path}: expected RollOutData")

    return sequences


def _split_agent_sequences(
    sequences: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_ratio: float,
    rng: random.Random,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    indices = list(range(len(sequences)))
    rng.shuffle(indices)

    if len(indices) <= 1 or val_ratio == 0.0:
        train_idx = indices
        val_idx: list[int] = []
    else:
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    return train_sequences, val_sequences


def build_datasets(
    data_dir: Path,
    val_ratio: float,
    history_len: int,
    future_len: int,
    decoder_context_len: int,
    window_stride: int,
    seed: int,
) -> tuple[OccupancyWindowDataset, OccupancyWindowDataset, DatasetStats]:
    rng = random.Random(seed)

    all_train_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    all_val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    for pt_file in pt_files:
        scene_sequences = _load_agent_sequences_from_file(
            pt_file,
            history_len=history_len,
            future_len=future_len,
            anchor_stride=window_stride,
        )
        train_seq, val_seq = _split_agent_sequences(scene_sequences, val_ratio, rng)
        all_train_sequences.extend(train_seq)
        all_val_sequences.extend(val_seq)

    num_train_anchors = int(len(all_train_sequences))
    num_val_anchors = int(len(all_val_sequences))

    train_dataset = OccupancyWindowDataset(
        all_train_sequences,
        history_len=history_len,
        future_len=future_len,
        decoder_context_len=decoder_context_len,
        stride=window_stride,
    )
    val_dataset = OccupancyWindowDataset(
        all_val_sequences,
        history_len=history_len,
        future_len=future_len,
        decoder_context_len=decoder_context_len,
        stride=window_stride,
    )

    stats = DatasetStats(
        num_scene_files=len(pt_files),
        num_train_agent_sequences=len(all_train_sequences),
        num_val_agent_sequences=len(all_val_sequences),
        num_train_anchors=num_train_anchors,
        num_val_anchors=num_val_anchors,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )
    return train_dataset, val_dataset, stats
