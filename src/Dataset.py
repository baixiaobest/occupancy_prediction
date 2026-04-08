from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.occupancy_patch import slice_centered_patch
from src.rollout_data import RollOutData, SceneRollOutData


def _build_window_centers(
    position_window: torch.Tensor,
    history_len: int,
    future_len: int,
) -> torch.Tensor:
    """Build per-frame crop centers for one window.

    All frames are pinned to the current timestep center so rollout context
    stays in the anchor ego frame.
    """
    window_size = int(history_len + future_len)
    centers = torch.as_tensor(position_window, dtype=torch.float32).clone()
    if centers.shape != (window_size, 2):
        raise ValueError("Position window must have shape (history_len+future_len, 2)")

    current_center = centers[history_len].clone()
    centers[:] = current_center
    return centers


def _compute_position_offsets(
    position_window: torch.Tensor,
    history_len: int,
    future_len: int,
) -> torch.Tensor:
    """Compute per-frame offsets from the anchor center in world coordinates."""
    window_size = int(history_len + future_len)
    positions = torch.as_tensor(position_window, dtype=torch.float32)
    if positions.shape != (window_size, 2):
        raise ValueError("position_window must have shape (history_len+future_len, 2)")
    anchor_position = positions[int(history_len)]
    return positions - anchor_position


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
    """Reference record for one fixed-center lazy sample.

    The dynamic/static global maps are shared across many refs. Local windows are
    sliced in `__getitem__` using the anchor ego center for the full window.
    """

    dynamic_global: torch.Tensor
    static_global: torch.Tensor
    position_global: torch.Tensor
    velocity_window: torch.Tensor
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
        static_seq: torch.Tensor,
        velocity_window: torch.Tensor,
        position_window: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if seq.ndim != 3:
            raise ValueError("Sequence must have shape (history+future, H, W)")
        if seq.shape[0] != self.window_size:
            raise ValueError("Sequence length must be history_len + future_len")
        if static_seq.ndim != 3:
            raise ValueError("Static sequence must have shape (history+future, H, W)")
        if static_seq.shape[0] != self.window_size:
            raise ValueError("Static sequence length must be history_len + future_len")
        if static_seq.shape[-2:] != seq.shape[-2:]:
            raise ValueError("Static sequence shape must match sequence spatial shape")

        vel_window = torch.as_tensor(velocity_window, dtype=torch.float32)
        if vel_window.shape == (2,):
            vel_window = vel_window.unsqueeze(0).repeat(self.window_size, 1)
        if vel_window.shape != (self.window_size, 2):
            raise ValueError("Velocity window must have shape (history_len+future_len, 2) or (2,)")

        past = seq[: self.history_len]
        future = seq[self.history_len : self.window_size]
        current_velocity = vel_window[self.history_len]
        future_velocities = vel_window[self.history_len : self.window_size]

        position_offsets = _compute_position_offsets(
            position_window=position_window,
            history_len=self.history_len,
            future_len=self.future_len,
        )
        future_position_offsets = position_offsets[self.history_len : self.window_size]

        x_encoder_dynamic = torch.cat([past, future], dim=0).unsqueeze(0)
        x_decoder_dynamic = past[-self.decoder_context_len :].unsqueeze(0)
        x_static = static_seq.unsqueeze(0)
        y = future.unsqueeze(0)
        return (
            x_encoder_dynamic,
            x_decoder_dynamic,
            x_static,
            current_velocity,
            future_velocities,
            future_position_offsets,
            y,
        )

    def _centers_from_motion(
        self,
        *,
        position_window: torch.Tensor,
    ) -> torch.Tensor:
        return _build_window_centers(
            position_window=position_window,
            history_len=self.history_len,
            future_len=self.future_len,
        )

class OccupancyWindowDataset(_OccupancyWindowBase):
    """Window dataset over precomputed local occupancy windows."""

    def __init__(
        self,
        sequences: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
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

        for seq, static_seq, velocity, position_window in self.samples:
            if seq.ndim != 3:
                raise ValueError("Each sequence must have shape (history+future, H, W)")
            if seq.shape[0] != self.window_size:
                raise ValueError("Each sequence must have exactly history_len + future_len frames")
            if static_seq.ndim != 3:
                raise ValueError("Each static sequence must have shape (history+future, H, W)")
            if static_seq.shape[0] != self.window_size:
                raise ValueError("Each static sequence must have exactly history_len + future_len frames")
            if static_seq.shape[-2:] != seq.shape[-2:]:
                raise ValueError("Static sequence spatial shape must match sequence shape")
            vel = torch.as_tensor(velocity, dtype=torch.float32)
            if vel.shape != (2,) and vel.shape != (self.window_size, 2):
                raise ValueError("Each velocity must have shape (2,) or (history_len+future_len, 2)")
            pos = torch.as_tensor(position_window, dtype=torch.float32)
            if pos.shape != (self.window_size, 2):
                raise ValueError("Each position_window must have shape (history_len+future_len, 2)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        seq, static_seq, velocity_window, position_window = self.samples[index]
        return self._format_model_io(seq, static_seq, velocity_window, position_window)


class LazyOccupancyWindowDataset(_OccupancyWindowBase):
    """Window dataset that slices local occupancy on demand in __getitem__."""

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

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        ref = self.sample_refs[index]

        position_window = torch.as_tensor(ref.position_global[int(ref.start_t) : int(ref.end_t)], dtype=torch.float32)

        centers_window = self._centers_from_motion(
            position_window=position_window,
        )

        dynamic_frames: list[torch.Tensor] = []
        static_frames: list[torch.Tensor] = []
        for local_t, absolute_t in enumerate(range(int(ref.start_t), int(ref.end_t))):
            center_xy = centers_window[local_t]
            dynamic_frames.append(
                slice_centered_patch(
                    ref.dynamic_global[absolute_t],
                    center_xy,
                    ref.scene_origin,
                    ref.resolution_xy,
                    ref.patch_shape,
                    binary=True,
                    prefer_view=True,
                )
            )
            static_frames.append(
                slice_centered_patch(
                    ref.static_global,
                    center_xy,
                    ref.scene_origin,
                    ref.resolution_xy,
                    ref.patch_shape,
                    binary=True,
                    prefer_view=True,
                )
            )
        seq = torch.stack(dynamic_frames, dim=0)
        static_seq = torch.stack(static_frames, dim=0)
        return self._format_model_io(seq, static_seq, ref.velocity_window, position_window)


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
    ) -> tuple[
        torch.Tensor,
        tuple[float, float],
        tuple[float, float],
        tuple[int, int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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

        scene_velocity_data = getattr(scene, "scene_velocity_trajectories", None)
        if scene_velocity_data is None:
            raise ValueError("scene_velocity_trajectories is required")

        scene_velocity_trajectories = torch.as_tensor(scene_velocity_data, dtype=torch.float32)
        if scene_velocity_trajectories.ndim != 3:
            raise ValueError("scene_velocity_trajectories must have shape (num_agents, total_time, 2)")
        expected_shape = (int(dynamic_maps.shape[0]), int(dynamic_maps.shape[1]), 2)
        if tuple(scene_velocity_trajectories.shape) != expected_shape:
            raise ValueError("scene_velocity_trajectories must match dynamic_maps agent/time dimensions")

        scene_position_data = getattr(scene, "scene_position_trajectories", None)
        if scene_position_data is None:
            raise ValueError("scene_position_trajectories is required")
        scene_position_trajectories = torch.as_tensor(scene_position_data, dtype=torch.float32)
        if tuple(scene_position_trajectories.shape) != expected_shape:
            raise ValueError("scene_position_trajectories must match dynamic_maps agent/time dimensions")

        return (
            scene_static,
            scene_origin,
            resolution_xy,
            patch_shape,
            dynamic_maps,
            scene_velocity_trajectories,
            scene_position_trajectories,
        )

    def _iter_valid_windows(
        self,
        dynamic_maps: torch.Tensor,
        scene_velocity_trajectories: torch.Tensor,
        scene_position_trajectories: torch.Tensor,
    ):
        """Yield validated per-window metadata for each agent in one scene."""
        window_size = int(self.history_len + self.future_len)
        if window_size <= 0:
            return

        step = max(1, self.window_stride)
        total_time = int(dynamic_maps.shape[1])
        max_start = total_time - window_size
        if max_start < 0:
            return

        for agent_idx in range(int(dynamic_maps.shape[0])):
            agent_dynamic = dynamic_maps[agent_idx]
            position_global = scene_position_trajectories[agent_idx]

            for start_t in range(0, max_start + 1, step):
                end_t = int(start_t + window_size)
                velocity_window = scene_velocity_trajectories[agent_idx, start_t:end_t].to(dtype=torch.float32)

                yield (
                    agent_dynamic,
                    position_global,
                    velocity_window,
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

    def _load_agent_sequences_from_file(self, pt_path: Path) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            payload = torch.load(pt_path, map_location="cpu")
        except AttributeError as exc:
            # Legacy pickle payloads reference removed classes (e.g., AnchorRollOutData).
            raise ValueError(
                f"Failed to load {pt_path}: legacy rollout format is no longer supported. "
                "Regenerate rollout .pt files using the current scripts/ORCA_rollout.py format."
            ) from exc

        def _from_scene_obj(scene: SceneRollOutData) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            (
                scene_static,
                scene_origin,
                resolution_xy,
                patch_shape,
                dynamic_maps,
                scene_velocity_trajectories,
                scene_position_trajectories,
            ) = self._unpack_scene_maps(scene)

            window_size = int(self.history_len + self.future_len)

            for (
                agent_dynamic,
                position_global,
                velocity_window,
                start_t,
                end_t,
            ) in self._iter_valid_windows(
                dynamic_maps,
                scene_velocity_trajectories,
                scene_position_trajectories,
            ):
                position_window = torch.as_tensor(position_global[start_t:end_t], dtype=torch.float32)

                centers_window = _build_window_centers(
                    position_window=position_window,
                    history_len=self.history_len,
                    future_len=self.future_len,
                )

                dynamic_window: list[torch.Tensor] = []
                static_window: list[torch.Tensor] = []
                for local_t, absolute_t in enumerate(range(start_t, end_t)):
                    center_xy = centers_window[local_t]
                    dynamic_global_t = self._to_2d_binary(agent_dynamic[absolute_t])
                    dynamic_local_t = slice_centered_patch(
                        grid=dynamic_global_t,
                        center_xy=center_xy,
                        origin_xy=scene_origin,
                        resolution_xy=resolution_xy,
                        patch_shape=patch_shape,
                        binary=True,
                        prefer_view=False,
                    )
                    dynamic_window.append(dynamic_local_t)
                    static_local_t = slice_centered_patch(
                        grid=scene_static,
                        center_xy=center_xy,
                        origin_xy=scene_origin,
                        resolution_xy=resolution_xy,
                        patch_shape=patch_shape,
                        binary=True,
                        prefer_view=False,
                    )
                    static_window.append(static_local_t)

                if len(dynamic_window) != window_size:
                    continue

                out.append(
                    (
                        torch.stack(dynamic_window, dim=0),
                        torch.stack(static_window, dim=0),
                        velocity_window,
                        position_window,
                    )
                )

            return out

        sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if isinstance(payload, RollOutData):
            for scene in payload.scenes:
                sequences.extend(_from_scene_obj(scene))
        else:
            raise ValueError(f"Unsupported payload format in {pt_path}: expected RollOutData")

        return sequences

    def _load_lazy_sample_refs_from_file(self, pt_path: Path) -> list[LazySampleRef]:
        """Load window references without materializing local occupancy windows."""
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
            (
                scene_static,
                scene_origin,
                resolution_xy,
                patch_shape,
                dynamic_maps,
                scene_velocity_trajectories,
                scene_position_trajectories,
            ) = self._unpack_scene_maps(scene)
            for (
                agent_dynamic,
                position_global,
                velocity_window,
                start_t,
                end_t,
            ) in self._iter_valid_windows(
                dynamic_maps,
                scene_velocity_trajectories,
                scene_position_trajectories,
            ):
                refs.append(
                    LazySampleRef(
                        dynamic_global=agent_dynamic,
                        static_global=scene_static,
                        position_global=position_global,
                        velocity_window=velocity_window.clone(),
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
