from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RollOutData:
    """Stored rollout data containing static+dynamic occupancy and timestep length."""

    static_maps: List[torch.Tensor]
    dynamic_grids: List[List[torch.Tensor]]
    dt: float
