from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class SceneRollOutData:
    """Stored rollout payload for one scene.

    Compact hierarchy:
    - scene_static_map: one static map for the full scene canvas.
    - scene_dynamic_maps: global dynamic occupancy with shape
        (num_agents, total_time, H, W).
        `scene_dynamic_maps[i, t]` is the global occupancy map at absolute
        timestep `t` with centered agent `i` removed.
    - scene_velocity_trajectories: per-agent velocity with shape
        (num_agents, total_time, 2).
    - scene_position_trajectories: per-agent global position with shape
        (num_agents, total_time, 2).
    - local_map_shape: local crop size (H, W) used by dataset loading.
    """

    dt: float
    occupancy_resolution: Tuple[float, float]
    occupancy_origin: Tuple[float, float]
    frame_offsets: List[int]
    scene_static_map: torch.Tensor | None = None
    scene_dynamic_maps: torch.Tensor | None = None
    scene_velocity_trajectories: torch.Tensor | None = None
    scene_position_trajectories: torch.Tensor | None = None
    scene_map_origin: Tuple[float, float] | None = None
    local_map_shape: Tuple[int, int] | None = None


@dataclass
class RollOutData:
    """Top-level rollout payload containing multiple scenes."""

    scenes: List[SceneRollOutData] = field(default_factory=list)
