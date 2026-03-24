from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class AnchorRollOutData:
    """One anchor snapshot for a centered agent.

    Fields:
    - anchor_time: simulation timestep index used as anchor.
    - static_map: static occupancy map (H, W) centered at this anchor.
    - current_velocity: ego velocity [vx, vy] at anchor_time.
    - frames: occupancy frames over temporal offsets around the anchor.
      Each frame has shape (H, W).
    """

    anchor_time: int
    static_map: torch.Tensor
    current_velocity: torch.Tensor
    frames: List[torch.Tensor]


@dataclass
class AgentRollOutData:
    """All anchor snapshots for one centered agent.

    - anchors: map from anchor timestep -> anchor payload.
    """

    agent_index: int
    anchors: Dict[int, AnchorRollOutData] = field(default_factory=dict)


@dataclass
class RollOutData:
    """Stored rollout payload for one scene.

    Explicit hierarchy:
    scene -> agents[agent_index] -> anchors[anchor_time] -> frames[offset_idx]
    """

    dt: float
    occupancy_resolution: Tuple[float, float]
    occupancy_origin: Tuple[float, float]
    frame_offsets: List[int]
    agents: Dict[int, AgentRollOutData] = field(default_factory=dict)
