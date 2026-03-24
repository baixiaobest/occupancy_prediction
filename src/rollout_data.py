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
    current_velocities: torch.Tensor | None = None


@dataclass
class SceneRollOutData:
    """Stored rollout payload for one scene.

    Compact hierarchy:
    - scene_static_map: one static map for the full scene canvas.
    - scene_dynamic_maps[agent_index]: one dynamic map (H, W) per centered agent.
    - agents[agent_index] contains anchor_times + anchor_centers + current_velocities.
    - local_map_shape gives the map size (H, W) that should be sliced per anchor.
    """

    dt: float
    occupancy_resolution: Tuple[float, float]
    occupancy_origin: Tuple[float, float]
    frame_offsets: List[int]
    agents: Dict[int, AgentRollOutData] = field(default_factory=dict)
    scene_static_map: torch.Tensor | None = None
    scene_dynamic_maps: Dict[int, torch.Tensor] = field(default_factory=dict)
    scene_map_origin: Tuple[float, float] | None = None
    local_map_shape: Tuple[int, int] | None = None


@dataclass
class RollOutData:
    """Top-level rollout payload containing multiple scenes."""

    scenes: List[SceneRollOutData] = field(default_factory=list)
