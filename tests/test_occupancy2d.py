from __future__ import annotations

import pytest
import torch

from src.occupancy2d import Occupancy2d


def test_generate_cpu_with_center_offset() -> None:
    occupancy = Occupancy2d(
        resolution=(0.1, 0.1),
        size=(2.0, 2.0),
        trajectory=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        static_obstacles=[],
    )

    grids = occupancy.generate(center_offset=torch.tensor([0.5, -0.25], dtype=torch.float32))

    assert len(grids) == 1
    assert grids[0].device.type == "cpu"
    assert int(grids[0].sum().item()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_generate_cuda_trajectory_uses_cuda_device() -> None:
    occupancy = Occupancy2d(
        resolution=(0.1, 0.1),
        size=(2.0, 2.0),
        trajectory=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, device="cuda"),
        static_obstacles=[],
    )

    grids = occupancy.generate(center_offset=torch.tensor([0.5, -0.25], dtype=torch.float32, device="cuda"))

    assert len(grids) == 1
    assert grids[0].device.type == "cuda"
    assert int(grids[0].sum().item()) > 0