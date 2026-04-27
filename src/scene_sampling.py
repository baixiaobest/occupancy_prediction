from __future__ import annotations

import copy
import random
from typing import Callable

from src.scene import Scene


def make_scene_factory(
    scenes: list[Scene],
    *,
    selection: str,
    fixed_scene_index: int,
    seed: int,
) -> Callable[[], Scene]:
    """Build a deterministic scene sampler for random/cycle/fixed selection modes."""
    scene_count = len(scenes)
    if scene_count == 0:
        raise ValueError("scenes must not be empty")
    if selection not in {"random", "cycle", "fixed"}:
        raise ValueError("selection must be one of: random, cycle, fixed")

    rng = random.Random(int(seed))
    fixed_idx = int(fixed_scene_index) % scene_count
    next_idx = 0

    def factory() -> Scene:
        nonlocal next_idx
        if selection == "fixed":
            scene = scenes[fixed_idx]
        elif selection == "cycle":
            scene = scenes[next_idx % scene_count]
            next_idx += 1
        else:
            scene = scenes[rng.randrange(scene_count)]
        return copy.deepcopy(scene)

    return factory


__all__ = ["make_scene_factory"]
