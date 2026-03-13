from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RollOutData:
    """Stored rollout data containing occupancy tensors and timestep length."""

    occupancy_grids: List[List[torch.Tensor]]
    dt: float
