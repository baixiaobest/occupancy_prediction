from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.rollout_data import RollOutData


@dataclass
class DatasetStats:
    num_scene_files: int
    num_train_origins: int
    num_val_origins: int
    num_train_samples: int
    num_val_samples: int


class OccupancyWindowDataset(Dataset):
    """Sliding-window dataset over dynamic occupancy sequences."""

    def __init__(
        self,
        sequences: Sequence[tuple[torch.Tensor, torch.Tensor]],
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

        for seq_idx, (seq, static_map) in enumerate(self.sequences):
            if seq.ndim != 3:
                raise ValueError("Each sequence must have shape (T, H, W)")
            if static_map.ndim != 2:
                raise ValueError("Each static map must have shape (H, W)")
            if static_map.shape != seq.shape[-2:]:
                raise ValueError("Static map shape must match sequence spatial shape")
            t = seq.shape[0]
            if t < self.window_size:
                continue
            for start in range(0, t - self.window_size + 1, stride):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, start = self.samples[index]
        seq, static_map = self.sequences[seq_idx]

        past = seq[start : start + self.history_len]
        future = seq[start + self.history_len : start + self.window_size]

        x_encoder_dynamic = torch.cat([past, future], dim=0).unsqueeze(0)
        x_decoder_dynamic = past[-self.decoder_context_len :].unsqueeze(0)
        x_static = static_map.unsqueeze(0)
        y = future.unsqueeze(0)
        return x_encoder_dynamic, x_decoder_dynamic, x_static, y


def _load_scene_origins(pt_path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    payload = torch.load(pt_path, map_location="cpu")

    def _to_2d(frame: object) -> torch.Tensor:
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
        return (tensor > 0).float()

    def _from_rollout_obj(obj: RollOutData) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(obj, "dynamic_grids") and hasattr(obj, "static_maps"):
            if len(obj.dynamic_grids) != len(obj.static_maps):
                raise ValueError("dynamic_grids and static_maps must have same length")

            out: list[tuple[torch.Tensor, torch.Tensor]] = []
            for dyn_series, static_map in zip(obj.dynamic_grids, obj.static_maps):
                dyn_frames = [_to_2d(frame) for frame in dyn_series]
                if not dyn_frames:
                    continue
                dynamic_seq = torch.stack(dyn_frames, dim=0)
                out.append((dynamic_seq, _to_2d(static_map)))
            return out

        raise ValueError("Unsupported RollOutData schema")

    sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
    if isinstance(payload, RollOutData):
        sequences.extend(_from_rollout_obj(payload))
    elif isinstance(payload, dict):
        if "dynamic_grids" in payload and "static_maps" in payload:
            pseudo = RollOutData(
                static_maps=payload["static_maps"],
                dynamic_grids=payload["dynamic_grids"],
                dt=float(payload.get("dt", 0.0)),
            )
            sequences.extend(_from_rollout_obj(pseudo))
        else:
            raise ValueError(f"Unsupported payload format in {pt_path}")
    elif isinstance(payload, list):
        if all(isinstance(x, RollOutData) for x in payload):
            for item in payload:
                sequences.extend(_from_rollout_obj(item))
        else:
            raise ValueError(
                "List payload must contain RollOutData objects with dynamic_grids/static_maps"
            )
    else:
        raise ValueError(f"Unsupported payload format in {pt_path}")

    return sequences


def _split_origins(
    sequences: Sequence[tuple[torch.Tensor, torch.Tensor]],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
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

    all_train_sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
    all_val_sequences: list[tuple[torch.Tensor, torch.Tensor]] = []

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    for pt_file in pt_files:
        scene_sequences = _load_scene_origins(pt_file)
        train_seq, val_seq = _split_origins(scene_sequences, val_ratio, rng)
        all_train_sequences.extend(train_seq)
        all_val_sequences.extend(val_seq)

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
        num_train_origins=len(all_train_sequences),
        num_val_origins=len(all_val_sequences),
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )
    return train_dataset, val_dataset, stats
