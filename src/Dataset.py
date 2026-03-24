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
    """Sliding-window dataset over dynamic occupancy sequences."""

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
        self.sequences = list(sequences)
        self.samples: list[tuple[int, int]] = []

        for seq_idx, (seq, static_seq, velocity_seq) in enumerate(self.sequences):
            if seq.ndim != 3:
                raise ValueError("Each sequence must have shape (T, H, W)")
            if static_seq.ndim != 3:
                raise ValueError("Each static sequence must have shape (T, H, W)")
            if static_seq.shape[-2:] != seq.shape[-2:]:
                raise ValueError("Static sequence spatial shape must match dynamic sequence")
            if velocity_seq.ndim != 2 or velocity_seq.shape[1] != 2:
                raise ValueError("Each velocity sequence must have shape (T, 2)")
            if static_seq.shape[0] != seq.shape[0]:
                raise ValueError("Static sequence length must match dynamic sequence length")
            if velocity_seq.shape[0] != seq.shape[0]:
                raise ValueError("Velocity sequence length must match dynamic sequence length")
            t = seq.shape[0]
            if t < self.window_size:
                continue
            for start in range(0, t - self.window_size + 1, stride):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, start = self.samples[index]
        seq, static_seq, velocity_seq = self.sequences[seq_idx]

        past = seq[start : start + self.history_len]
        future = seq[start + self.history_len : start + self.window_size]
        # Anchor velocity corresponds to first prediction timestep.
        anchor_index = start + self.history_len
        current_velocity = velocity_seq[anchor_index]
        current_static = static_seq[anchor_index]

        x_encoder_dynamic = torch.cat([past, future], dim=0).unsqueeze(0)
        x_decoder_dynamic = past[-self.decoder_context_len :].unsqueeze(0)
        x_static = current_static.unsqueeze(0)
        y = future.unsqueeze(0)
        return x_encoder_dynamic, x_decoder_dynamic, x_static, current_velocity, y


def _load_agent_sequences_from_file(pt_path: Path) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        if not scene.scene_dynamic_maps:
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

        for agent_idx in sorted(scene.agents.keys()):
            agent_data = scene.agents[agent_idx]
            dynamic_map_obj = scene.scene_dynamic_maps.get(agent_idx)
            if dynamic_map_obj is None:
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
            dynamic_map = torch.as_tensor(dynamic_map_obj, dtype=torch.float32)
            if centers_tensor.shape != (len(anchor_times), 2):
                raise ValueError("anchor_centers must have shape (num_anchors, 2)")
            if velocity_tensor.shape != (len(anchor_times), 2):
                raise ValueError("current_velocities must have shape (num_anchors, 2)")
            if dynamic_map.ndim != 2:
                raise ValueError("scene_dynamic_maps[agent] must be a 2D map")

            anchor_grids: list[torch.Tensor] = []
            anchor_static_maps: list[torch.Tensor] = []
            anchor_velocities: list[torch.Tensor] = []
            for anchor_idx in range(len(anchor_times)):
                center_xy = centers_tensor[anchor_idx]
                dynamic_local = _slice_centered_patch(
                    grid=dynamic_map,
                    center_xy=center_xy,
                    origin_xy=scene_origin,
                    resolution_xy=resolution_xy,
                    patch_shape=patch_shape,
                )
                static_local = _slice_centered_patch(
                    grid=scene_static,
                    center_xy=center_xy,
                    origin_xy=scene_origin,
                    resolution_xy=resolution_xy,
                    patch_shape=patch_shape,
                )

                anchor_grids.append(dynamic_local)
                anchor_static_maps.append(static_local)
                anchor_velocities.append(velocity_tensor[anchor_idx])

            if not anchor_grids:
                continue

            dynamic_seq = torch.stack(anchor_grids, dim=0)
            static_seq = torch.stack(anchor_static_maps, dim=0)
            velocity_seq = torch.stack(anchor_velocities, dim=0).to(dtype=torch.float32)
            out.append((dynamic_seq, static_seq, velocity_seq))

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
        scene_sequences = _load_agent_sequences_from_file(pt_file)
        train_seq, val_seq = _split_agent_sequences(scene_sequences, val_ratio, rng)
        all_train_sequences.extend(train_seq)
        all_val_sequences.extend(val_seq)

    num_train_anchors = int(sum(seq.shape[0] for seq, _, _ in all_train_sequences))
    num_val_anchors = int(sum(seq.shape[0] for seq, _, _ in all_val_sequences))

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
