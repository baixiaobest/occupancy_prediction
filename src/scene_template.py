from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.scene import ObstacleSpec, PathSpec, RegionPairSpec, RegionSpec, Scene


class SceneTemplate(ABC):
    """Base class for scene templates that generate a list of scenes."""

    @abstractmethod
    def generate(self, num_levels: int) -> List[Scene]:
        """Generate a list of scenes for the requested number of levels."""

    @staticmethod
    def _linear_levels(value_range: Tuple[float, float], num_levels: int) -> List[float]:
        """Linearly sample values from `value_range` with `num_levels` samples."""
        if num_levels <= 0:
            raise ValueError("num_levels must be >= 1")

        start, end = float(value_range[0]), float(value_range[1])
        if num_levels == 1:
            return [start]

        step = (end - start) / float(num_levels - 1)
        return [start + idx * step for idx in range(num_levels)]


class StraightCorridorTemplate(SceneTemplate):
    """Template for straight-corridor scenes with one-way or opposing traffic.

    The corridor is aligned with +X from x=0 to x=length, centered around y=0.
    Width and length are linearly sampled by level from user-specified ranges.
    """

    def __init__(
        self,
        width_range: Tuple[float, float],
        length_range: Tuple[float, float],
        startup_agent_count_per_pair: int = 8,
        num_region_pairs: int = 1,
        spawn_depth_ratio: float = 0.15,
        spawn_width_ratio: float = 0.8,
        wall_thickness: float = 0.4,
    ) -> None:
        if startup_agent_count_per_pair < 0:
            raise ValueError("startup_agent_count_per_pair must be >= 0")
        if num_region_pairs not in (1, 2):
            raise ValueError("num_region_pairs must be 1 or 2")
        if not (0.0 < spawn_depth_ratio <= 0.5):
            raise ValueError("spawn_depth_ratio must be in (0, 0.5]")
        if not (0.0 < spawn_width_ratio <= 1.0):
            raise ValueError("spawn_width_ratio must be in (0, 1]")
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0")

        self.width_range = (float(width_range[0]), float(width_range[1]))
        self.length_range = (float(length_range[0]), float(length_range[1]))
        self.startup_agent_count_per_pair = int(startup_agent_count_per_pair)
        self.num_region_pairs = int(num_region_pairs)
        self.spawn_depth_ratio = float(spawn_depth_ratio)
        self.spawn_width_ratio = float(spawn_width_ratio)
        self.wall_thickness = float(wall_thickness)

    def generate(self, num_levels: int) -> List[Scene]:
        """Generate `num_levels` straight-corridor scenes.

        Level i uses linearly sampled width and length at the same interpolation ratio.
        """
        widths = self._linear_levels(self.width_range, num_levels)
        lengths = self._linear_levels(self.length_range, num_levels)

        scenes: List[Scene] = []
        for width, length in zip(widths, lengths):
            if width <= 0.0 or length <= 0.0:
                raise ValueError("Sampled width and length must be > 0")
            scenes.append(self._build_scene(width=width, length=length))

        return scenes

    def _build_scene(self, width: float, length: float) -> Scene:
        half_width = 0.5 * width
        wall_t = self.wall_thickness

        bottom_wall = ObstacleSpec(
            vertices=[
                (-wall_t, -half_width - wall_t),
                (length + wall_t, -half_width - wall_t),
                (length + wall_t, -half_width),
                (-wall_t, -half_width),
            ]
        )
        top_wall = ObstacleSpec(
            vertices=[
                (-wall_t, half_width),
                (length + wall_t, half_width),
                (length + wall_t, half_width + wall_t),
                (-wall_t, half_width + wall_t),
            ]
        )

        spawn_depth = max(1e-6, self.spawn_depth_ratio * length)
        spawn_half_width = max(1e-6, 0.5 * self.spawn_width_ratio * width)

        left_region = RegionSpec(
            min_corner=(0.0, -spawn_half_width),
            max_corner=(spawn_depth, spawn_half_width),
        )
        right_region = RegionSpec(
            min_corner=(length - spawn_depth, -spawn_half_width),
            max_corner=(length, spawn_half_width),
        )

        paths = [
            PathSpec(points=[(0.0, 0.0), (length, 0.0)]),
            PathSpec(points=[(length, 0.0), (0.0, 0.0)]),
        ]

        region_pairs = [
            RegionPairSpec(
                spawn_region=left_region,
                destination_region=right_region,
                startup_agent_count=self.startup_agent_count_per_pair,
                path_index=0,
            )
        ]

        if self.num_region_pairs == 2:
            region_pairs.append(
                RegionPairSpec(
                    spawn_region=right_region,
                    destination_region=left_region,
                    startup_agent_count=self.startup_agent_count_per_pair,
                    path_index=1,
                )
            )

        return Scene(
            agents=[],
            obstacles=[bottom_wall, top_wall],
            paths=paths,
            region_pairs=region_pairs,
        )
