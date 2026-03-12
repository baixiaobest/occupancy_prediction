from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Obstacle:
    """Axis-aligned box obstacle defined by center position and size."""

    position: torch.Tensor
    size: torch.Tensor

    def __post_init__(self) -> None:
        self.position = _to_tensor(self.position)
        self.size = _to_tensor(self.size)

    def update(self, position: Tuple[float, float] | torch.Tensor) -> None:
        self.position = _to_tensor(position, device=self.position.device)


def _to_tensor(
    value: Tuple[float, float] | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device) if device is not None else value
    return torch.tensor(value, dtype=torch.float32, device=device)
