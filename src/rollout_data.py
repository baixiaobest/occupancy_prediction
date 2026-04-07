from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class AgentRollOutData:
    """Compact per-agent metadata for one scene rollout."""

    agent_index: int
    anchor_times: List[int] = field(default_factory=list)
    anchor_centers: torch.Tensor | None = None


@dataclass
class SceneRollOutData:
    """Stored rollout payload for one scene.

        Compact hierarchy:
        - scene_static_map: one static map for the full scene canvas.
        - scene_dynamic_maps: global dynamic occupancy with shape
            (num_agents, total_time, H, W).
            `scene_dynamic_maps[i, t]` is the global occupancy map at absolute
            timestep `t` with centered agent `i` removed.
        - scene_velocity_trajectories stores per-agent velocity with shape
            (num_agents, total_time, 2).
        - agents[agent_index] contains anchor_times + anchor_centers.
        - local_map_shape gives the local crop size (H, W) used by dataset loading.
    """

    dt: float
    occupancy_resolution: Tuple[float, float]
    occupancy_origin: Tuple[float, float]
    frame_offsets: List[int]
    agents: Dict[int, AgentRollOutData] = field(default_factory=dict)
    scene_static_map: torch.Tensor | None = None
    scene_dynamic_maps: torch.Tensor | None = None
    scene_velocity_trajectories: torch.Tensor | None = None
    scene_map_origin: Tuple[float, float] | None = None
    local_map_shape: Tuple[int, int] | None = None


@dataclass
class RollOutData:
    """Top-level rollout payload containing multiple scenes."""

    scenes: List[SceneRollOutData] = field(default_factory=list)
