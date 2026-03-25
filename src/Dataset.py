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


@dataclass
class LazySampleRef:
    """Reference record for one anchor-centered lazy sample.

    The dynamic/static global maps are shared across many refs. Local windows are
    sliced in `__getitem__` using the fixed anchor center.
    """

    dynamic_global: torch.Tensor
    static_global: torch.Tensor
    center_xy: torch.Tensor
    current_velocity: torch.Tensor
    scene_origin: tuple[float, float]
    resolution_xy: tuple[float, float]
    patch_shape: tuple[int, int]
    start_t: int
    end_t: int


class _OccupancyWindowBase(Dataset):
    """Shared window formatting for eager and lazy occupancy datasets."""

    def __init__(
        self,
        history_len: int,
        future_len: int,
        decoder_context_len: int,
        stride: int,
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

    def _format_model_io(
        self,
        seq: torch.Tensor,
        current_static: torch.Tensor,
        current_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq.ndim != 3:
            raise ValueError("Sequence must have shape (history+future, H, W)")
        if seq.shape[0] != self.window_size:
            raise ValueError("Sequence length must be history_len + future_len")
        if current_static.ndim != 2:
            raise ValueError("Static map must have shape (H, W)")
        if current_static.shape != seq.shape[-2:]:
            raise ValueError("Static map shape must match sequence spatial shape")
        vel = torch.as_tensor(current_velocity, dtype=torch.float32)
        if vel.shape != (2,):
            raise ValueError("Current velocity must have shape (2,)")

        past = seq[: self.history_len]
        future = seq[self.history_len : self.window_size]

        x_encoder_dynamic = torch.cat([past, future], dim=0).unsqueeze(0)
        x_decoder_dynamic = past[-self.decoder_context_len :].unsqueeze(0)
        x_static = current_static.unsqueeze(0)
        y = future.unsqueeze(0)
        return x_encoder_dynamic, x_decoder_dynamic, x_static, vel, y

class OccupancyWindowDataset(_OccupancyWindowBase):
    """Anchor-window dataset over precomputed local occupancy windows."""

    def __init__(
        self,
        sequences: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        history_len: int = 16,
        future_len: int = 8,
        decoder_context_len: int = 8,
        stride: int = 1,
    ) -> None:
        super().__init__(
            history_len=history_len,
            future_len=future_len,
            decoder_context_len=decoder_context_len,
            stride=stride,
        )
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
        return self._format_model_io(seq, current_static, current_velocity)


class LazyOccupancyWindowDataset(_OccupancyWindowBase):
    """Anchor-window dataset that slices local occupancy on demand in __getitem__."""

    def __init__(
        self,
        sample_refs: Sequence[LazySampleRef],
        history_len: int = 16,
        future_len: int = 8,
        decoder_context_len: int = 8,
        stride: int = 1,
    ) -> None:
        super().__init__(
            history_len=history_len,
            future_len=future_len,
            decoder_context_len=decoder_context_len,
            stride=stride,
        )
        self.sample_refs = list(sample_refs)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.sample_refs[index]

        dynamic_frames: list[torch.Tensor] = []
        for absolute_t in range(int(ref.start_t), int(ref.end_t)):
            dynamic_frames.append(
                _slice_centered_patch_binary(
                    ref.dynamic_global[absolute_t],
                    ref.center_xy,
                    ref.scene_origin,
                    ref.resolution_xy,
                    ref.patch_shape,
                    prefer_view=True,
                )
            )
        seq = torch.stack(dynamic_frames, dim=0)
        static_local = _slice_centered_patch_binary(
            ref.static_global,
            ref.center_xy,
            ref.scene_origin,
            ref.resolution_xy,
            ref.patch_shape,
            prefer_view=True,
        )
        return self._format_model_io(seq, static_local, ref.current_velocity)


def _slice_centered_patch_binary(
    grid: torch.Tensor,
    center_xy: torch.Tensor,
    origin_xy: tuple[float, float],
    resolution_xy: tuple[float, float],
    patch_shape: tuple[int, int],
    *,
    prefer_view: bool,
) -> torch.Tensor:
    """Slice local patch as binary float tensor.

    When `prefer_view` is True and the full patch is in-bounds, return a view.
    Otherwise return a zero-padded copy.
    """
    patch_h, patch_w = patch_shape
    res_x, res_y = float(resolution_xy[0]), float(resolution_xy[1])
    center_x = float(center_xy[0].item())
    center_y = float(center_xy[1].item())
    half_w = 0.5 * patch_w * res_x
    half_h = 0.5 * patch_h * res_y

    start_x = int(np.floor((center_x - half_w - origin_xy[0]) / res_x))
    start_y = int(np.floor((center_y - half_h - origin_xy[1]) / res_y))
    end_x = start_x + patch_w
    end_y = start_y + patch_h

    if prefer_view and start_x >= 0 and start_y >= 0 and end_x <= grid.shape[1] and end_y <= grid.shape[0]:
        return (grid[start_y:end_y, start_x:end_x] > 0).float()

    out = torch.zeros((patch_h, patch_w), dtype=torch.float32)
    src_x0 = max(0, start_x)
    src_y0 = max(0, start_y)
    src_x1 = min(grid.shape[1], end_x)
    src_y1 = min(grid.shape[0], end_y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - start_x
    dst_y0 = src_y0 - start_y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = grid[src_y0:src_y1, src_x0:src_x1]
    return (out > 0).float()


class DatasetBuilder:
    """Build eager or lazy occupancy datasets from rollout files."""

    def __init__(
        self,
        data_dir: Path,
        val_ratio: float,
        history_len: int,
        future_len: int,
        decoder_context_len: int,
        window_stride: int,
        seed: int,
    ) -> None:
        self.data_dir = data_dir
        self.val_ratio = float(val_ratio)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.decoder_context_len = int(decoder_context_len)
        self.window_stride = int(window_stride)
        self.rng = random.Random(seed)

    @staticmethod
    def _to_2d_binary(frame: object) -> torch.Tensor:
        """Convert occupancy frame to 2D binary float tensor."""
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
        return (tensor > 0).float()

    def _unpack_scene_maps(
        self,
        scene: SceneRollOutData,
    ) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float], tuple[int, int], torch.Tensor]:
        """Validate and unpack common scene-level map fields used by loaders."""

        scene_static = self._to_2d_binary(scene.scene_static_map)
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

        return scene_static, scene_origin, resolution_xy, patch_shape, dynamic_maps

    def _iter_valid_anchors(self, scene: SceneRollOutData, dynamic_maps: torch.Tensor):
        """Yield validated per-anchor metadata for each agent in one scene."""
        for agent_idx in sorted(scene.agents.keys()):
            if agent_idx < 0 or agent_idx >= dynamic_maps.shape[0]:
                continue

            agent_data = scene.agents[agent_idx]
            anchor_times = agent_data.anchor_times
            if not anchor_times:
                continue
            if agent_data.anchor_centers is None or agent_data.current_velocities is None:
                raise ValueError("Agent metadata requires anchor_centers and current_velocities")

            centers_tensor = torch.as_tensor(agent_data.anchor_centers, dtype=torch.float32)
            velocity_tensor = torch.as_tensor(agent_data.current_velocities, dtype=torch.float32)
            if centers_tensor.shape != (len(anchor_times), 2):
                raise ValueError("anchor_centers must have shape (num_anchors, 2)")
            if velocity_tensor.shape != (len(anchor_times), 2):
                raise ValueError("current_velocities must have shape (num_anchors, 2)")

            agent_dynamic = dynamic_maps[agent_idx]
            total_time = int(agent_dynamic.shape[0])
            anchor_indices = list(range(0, len(anchor_times), max(1, self.window_stride)))
            for anchor_meta_idx in anchor_indices:
                anchor_t = int(anchor_times[anchor_meta_idx])
                start_t = anchor_t - self.history_len
                end_t = anchor_t + self.future_len
                if start_t < 0 or end_t > total_time:
                    continue
                yield (
                    agent_dynamic,
                    centers_tensor[anchor_meta_idx],
                    velocity_tensor[anchor_meta_idx].to(dtype=torch.float32),
                    int(start_t),
                    int(end_t),
                )

    @staticmethod
    def _split_samples(
        samples: Sequence[object],
        val_ratio: float,
        rng: random.Random,
    ) -> tuple[list[object], list[object]]:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0.0, 1.0)")

        indices = list(range(len(samples)))
        rng.shuffle(indices)

        if len(indices) <= 1 or val_ratio == 0.0:
            train_idx = indices
            val_idx: list[int] = []
        else:
            val_count = max(1, int(round(len(indices) * val_ratio)))
            val_count = min(val_count, len(indices) - 1)
            val_idx = indices[:val_count]
            train_idx = indices[val_count:]

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        return train_samples, val_samples

    def _load_agent_sequences_from_file(self, pt_path: Path) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            payload = torch.load(pt_path, map_location="cpu")
        except AttributeError as exc:
            # Legacy pickle payloads reference removed classes (e.g., AnchorRollOutData).
            raise ValueError(
                f"Failed to load {pt_path}: legacy rollout format is no longer supported. "
                "Regenerate rollout .pt files using the current scripts/ORCA_rollout.py format."
            ) from exc

        def _from_scene_obj(scene: SceneRollOutData) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            scene_static, scene_origin, resolution_xy, patch_shape, dynamic_maps = self._unpack_scene_maps(scene)

            window_size = int(self.history_len + self.future_len)

            for agent_dynamic, center_xy, velocity_xy, start_t, end_t in self._iter_valid_anchors(scene, dynamic_maps):
                static_local = _slice_centered_patch_binary(
                    grid=scene_static,
                    center_xy=center_xy,
                    origin_xy=scene_origin,
                    resolution_xy=resolution_xy,
                    patch_shape=patch_shape,
                    prefer_view=False,
                )

                dynamic_window: list[torch.Tensor] = []
                for absolute_t in range(start_t, end_t):
                    dynamic_global_t = self._to_2d_binary(agent_dynamic[absolute_t])
                    dynamic_local_t = _slice_centered_patch_binary(
                        grid=dynamic_global_t,
                        center_xy=center_xy,
                        origin_xy=scene_origin,
                        resolution_xy=resolution_xy,
                        patch_shape=patch_shape,
                        prefer_view=False,
                    )
                    dynamic_window.append(dynamic_local_t)

                if len(dynamic_window) != window_size:
                    continue

                out.append(
                    (
                        torch.stack(dynamic_window, dim=0),
                        static_local,
                        velocity_xy,
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

    def _load_lazy_sample_refs_from_file(self, pt_path: Path) -> list[LazySampleRef]:
        """Load anchor references without materializing local occupancy windows."""
        try:
            payload = torch.load(pt_path, map_location="cpu")
        except AttributeError as exc:
            raise ValueError(
                f"Failed to load {pt_path}: legacy rollout format is no longer supported. "
                "Regenerate rollout .pt files using the current scripts/ORCA_rollout.py format."
            ) from exc

        refs: list[LazySampleRef] = []
        if not isinstance(payload, RollOutData):
            raise ValueError(f"Unsupported payload format in {pt_path}: expected RollOutData")

        for scene in payload.scenes:
            scene_static, scene_origin, resolution_xy, patch_shape, dynamic_maps = self._unpack_scene_maps(scene)
            for agent_dynamic, center_xy, velocity_xy, start_t, end_t in self._iter_valid_anchors(scene, dynamic_maps):
                refs.append(
                    LazySampleRef(
                        dynamic_global=agent_dynamic,
                        static_global=scene_static,
                        center_xy=center_xy.clone(),
                        current_velocity=velocity_xy.clone(),
                        scene_origin=scene_origin,
                        resolution_xy=resolution_xy,
                        patch_shape=patch_shape,
                        start_t=start_t,
                        end_t=end_t,
                    )
                )

        return refs

    def _list_rollout_files(self) -> list[Path]:
        """Return sorted rollout file list and validate directory is non-empty."""
        pt_files = sorted(self.data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")
        return pt_files

    def _collect_split_samples(self, *, pt_files: Sequence[Path], loader) -> tuple[list[object], list[object]]:
        """Load all files with `loader` and split samples into train/val lists."""
        train_samples: list[object] = []
        val_samples: list[object] = []

        for pt_file in pt_files:
            samples = loader(pt_file)
            train_split, val_split = self._split_samples(samples, self.val_ratio, self.rng)
            train_samples.extend(train_split)
            val_samples.extend(val_split)

        return train_samples, val_samples

    @staticmethod
    def _build_stats(
        *,
        num_scene_files: int,
        train_items: Sequence[object],
        val_items: Sequence[object],
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> DatasetStats:
        """Construct DatasetStats shared by eager and lazy build flows."""
        return DatasetStats(
            num_scene_files=num_scene_files,
            num_train_agent_sequences=len(train_items),
            num_val_agent_sequences=len(val_items),
            num_train_anchors=len(train_items),
            num_val_anchors=len(val_items),
            num_train_samples=len(train_dataset),
            num_val_samples=len(val_dataset),
        )

    def build(self, *, lazy: bool = False) -> tuple[Dataset, Dataset, DatasetStats]:
        """Unified dataset build entry for eager or lazy occupancy slicing."""
        pt_files = self._list_rollout_files()
        dataset_cls = LazyOccupancyWindowDataset if lazy else OccupancyWindowDataset

        if lazy:
            train_refs, val_refs = self._collect_split_samples(
                pt_files=pt_files,
                loader=self._load_lazy_sample_refs_from_file,
            )
            train_dataset = dataset_cls(
                train_refs,
                history_len=self.history_len,
                future_len=self.future_len,
                decoder_context_len=self.decoder_context_len,
                stride=self.window_stride,
            )
            val_dataset = dataset_cls(
                val_refs,
                history_len=self.history_len,
                future_len=self.future_len,
                decoder_context_len=self.decoder_context_len,
                stride=self.window_stride,
            )
            stats = self._build_stats(
                num_scene_files=len(pt_files),
                train_items=train_refs,
                val_items=val_refs,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            return train_dataset, val_dataset, stats

        train_sequences, val_sequences = self._collect_split_samples(
            pt_files=pt_files,
            loader=self._load_agent_sequences_from_file,
        )
        train_dataset = dataset_cls(
            train_sequences,
            history_len=self.history_len,
            future_len=self.future_len,
            decoder_context_len=self.decoder_context_len,
            stride=self.window_stride,
        )
        val_dataset = dataset_cls(
            val_sequences,
            history_len=self.history_len,
            future_len=self.future_len,
            decoder_context_len=self.decoder_context_len,
            stride=self.window_stride,
        )
        stats = self._build_stats(
            num_scene_files=len(pt_files),
            train_items=train_sequences,
            val_items=val_sequences,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        return train_dataset, val_dataset, stats


def build_datasets(
    data_dir: Path,
    val_ratio: float,
    history_len: int,
    future_len: int,
    decoder_context_len: int,
    window_stride: int,
    seed: int,
    *,
    lazy: bool = False,
) -> tuple[Dataset, Dataset, DatasetStats]:
    """Build train/val datasets; set lazy=True for on-the-fly slicing."""
    builder = DatasetBuilder(
        data_dir=data_dir,
        val_ratio=val_ratio,
        history_len=history_len,
        future_len=future_len,
        decoder_context_len=decoder_context_len,
        window_stride=window_stride,
        seed=seed,
    )
    return builder.build(lazy=lazy)
