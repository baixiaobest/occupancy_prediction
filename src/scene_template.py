from __future__ import annotations

from abc import ABC, abstractmethod
import math
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
        spawn_density_range: Tuple[float, float] = (1.0, 1.0),
        spawn_velocity_range: Tuple[float, float] = (1.5, 1.5),
        num_region_pairs: int = 1,
        spawn_depth_ratio: float = 0.15,
        spawn_width_ratio: float = 0.8,
        wall_thickness: float = 0.4,
    ) -> None:
        if num_region_pairs not in (1, 2):
            raise ValueError("num_region_pairs must be 1 or 2")
        if spawn_density_range[0] < 0.0 or spawn_density_range[1] < 0.0:
            raise ValueError("spawn_density_range values must be >= 0")
        if spawn_velocity_range[0] < 0.0 or spawn_velocity_range[1] < 0.0:
            raise ValueError("spawn_velocity_range values must be >= 0")
        if not (0.0 < spawn_depth_ratio <= 0.5):
            raise ValueError("spawn_depth_ratio must be in (0, 0.5]")
        if not (0.0 < spawn_width_ratio <= 1.0):
            raise ValueError("spawn_width_ratio must be in (0, 1]")
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0")

        self.width_range = (float(width_range[0]), float(width_range[1]))
        self.length_range = (float(length_range[0]), float(length_range[1]))
        self.spawn_density_range = (
            float(spawn_density_range[0]),
            float(spawn_density_range[1]),
        )
        self.spawn_velocity_range = (
            float(spawn_velocity_range[0]),
            float(spawn_velocity_range[1]),
        )
        self.num_region_pairs = int(num_region_pairs)
        self.spawn_depth_ratio = float(spawn_depth_ratio)
        self.spawn_width_ratio = float(spawn_width_ratio)
        self.wall_thickness = float(wall_thickness)

    def generate(self, num_levels: int) -> List[Scene]:
        """Generate `num_levels` straight-corridor scenes.

        Level i uses linearly sampled width, length, and spawn density.
        Spawn count per region pair is floor(density * spawn_area).
        """
        widths = self._linear_levels(self.width_range, num_levels)
        lengths = self._linear_levels(self.length_range, num_levels)
        densities = self._linear_levels(self.spawn_density_range, num_levels)

        scenes: List[Scene] = []
        for width, length, density in zip(widths, lengths, densities):
            if width <= 0.0 or length <= 0.0:
                raise ValueError("Sampled width and length must be > 0")
            scenes.append(self._build_scene(width=width, length=length, spawn_density=density))

        return scenes

    def _build_scene(self, width: float, length: float, spawn_density: float) -> Scene:
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
        spawn_area = 2.0 * spawn_half_width * spawn_depth
        startup_agent_count = max(0, math.floor(spawn_density * spawn_area))

        left_region = RegionSpec(
            min_corner=(0.0, -spawn_half_width),
            max_corner=(spawn_depth, spawn_half_width),
        )
        right_region = RegionSpec(
            min_corner=(length - spawn_depth, -spawn_half_width),
            max_corner=(length, spawn_half_width),
        )

        region_pairs = [
            RegionPairSpec(
                spawn_region=left_region,
                destination_region=right_region,
                startup_agent_count=startup_agent_count,
                path_index=None,
                velocity_range=self.spawn_velocity_range,
            )
        ]

        if self.num_region_pairs == 2:
            region_pairs.append(
                RegionPairSpec(
                    spawn_region=right_region,
                    destination_region=left_region,
                    startup_agent_count=startup_agent_count,
                    path_index=None,
                    velocity_range=self.spawn_velocity_range,
                )
            )

        return Scene(
            agents=[],
            obstacles=[bottom_wall, top_wall],
            paths=[],
            region_pairs=region_pairs,
        )


class LShapeCorridorTemplate(SceneTemplate):
    """Template for L-shaped corridor scenes with one-way or opposing traffic.

    The centerline runs from `(-horizontal_length, 0)` to `(0, 0)` to `(0, vertical_length)`.
    Corridor width is constant for both legs.
    """

    def __init__(
        self,
        width_range: Tuple[float, float],
        horizontal_length_range: Tuple[float, float],
        vertical_length_range: Tuple[float, float],
        spawn_density_range: Tuple[float, float] = (1.0, 1.0),
        spawn_velocity_range: Tuple[float, float] = (1.5, 1.5),
        num_region_pairs: int = 1,
        spawn_depth_ratio: float = 0.15,
        spawn_width_ratio: float = 0.8,
        wall_thickness: float = 0.4,
        turn_smoothing_segments: int = 6,
        turn_radius_ratio: float = 0.6,
    ) -> None:
        if num_region_pairs not in (1, 2):
            raise ValueError("num_region_pairs must be 1 or 2")
        if spawn_density_range[0] < 0.0 or spawn_density_range[1] < 0.0:
            raise ValueError("spawn_density_range values must be >= 0")
        if spawn_velocity_range[0] < 0.0 or spawn_velocity_range[1] < 0.0:
            raise ValueError("spawn_velocity_range values must be >= 0")
        if not (0.0 < spawn_depth_ratio <= 0.5):
            raise ValueError("spawn_depth_ratio must be in (0, 0.5]")
        if not (0.0 < spawn_width_ratio <= 1.0):
            raise ValueError("spawn_width_ratio must be in (0, 1]")
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0")
        if turn_smoothing_segments < 1:
            raise ValueError("turn_smoothing_segments must be >= 1")
        if turn_radius_ratio <= 0.0:
            raise ValueError("turn_radius_ratio must be > 0")

        self.width_range = (float(width_range[0]), float(width_range[1]))
        self.horizontal_length_range = (
            float(horizontal_length_range[0]),
            float(horizontal_length_range[1]),
        )
        self.vertical_length_range = (
            float(vertical_length_range[0]),
            float(vertical_length_range[1]),
        )
        self.spawn_density_range = (
            float(spawn_density_range[0]),
            float(spawn_density_range[1]),
        )
        self.spawn_velocity_range = (
            float(spawn_velocity_range[0]),
            float(spawn_velocity_range[1]),
        )
        self.num_region_pairs = int(num_region_pairs)
        self.spawn_depth_ratio = float(spawn_depth_ratio)
        self.spawn_width_ratio = float(spawn_width_ratio)
        self.wall_thickness = float(wall_thickness)
        self.turn_smoothing_segments = int(turn_smoothing_segments)
        self.turn_radius_ratio = float(turn_radius_ratio)

    def generate(self, num_levels: int) -> List[Scene]:
        """Generate `num_levels` L-shaped corridor scenes.

        Level i uses linearly sampled width, horizontal length, vertical length, and density.
        Startup agent counts are floor(density * spawn_area).
        """
        widths = self._linear_levels(self.width_range, num_levels)
        horizontal_lengths = self._linear_levels(self.horizontal_length_range, num_levels)
        vertical_lengths = self._linear_levels(self.vertical_length_range, num_levels)
        densities = self._linear_levels(self.spawn_density_range, num_levels)

        scenes: List[Scene] = []
        for width, horizontal_length, vertical_length, density in zip(
            widths, horizontal_lengths, vertical_lengths, densities
        ):
            if width <= 0.0 or horizontal_length <= 0.0 or vertical_length <= 0.0:
                raise ValueError("Sampled width and lengths must be > 0")
            scenes.append(
                self._build_scene(
                    width=width,
                    horizontal_length=horizontal_length,
                    vertical_length=vertical_length,
                    spawn_density=density,
                )
            )

        return scenes

    def _build_scene(
        self,
        width: float,
        horizontal_length: float,
        vertical_length: float,
        spawn_density: float,
    ) -> Scene:
        half_width = 0.5 * width
        wall_t = self.wall_thickness

        left_x = -horizontal_length
        top_y = vertical_length

        lower_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, -half_width - wall_t),
                (half_width + wall_t, -half_width - wall_t),
                (half_width + wall_t, -half_width),
                (left_x - wall_t, -half_width),
            ]
        )
        lower_inner_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, half_width),
                (-half_width, half_width),
                (-half_width, half_width + wall_t),
                (left_x - wall_t, half_width + wall_t),
            ]
        )
        right_wall = ObstacleSpec(
            vertices=[
                (half_width, -half_width - wall_t),
                (half_width + wall_t, -half_width - wall_t),
                (half_width + wall_t, top_y + wall_t),
                (half_width, top_y + wall_t),
            ]
        )
        right_inner_wall = ObstacleSpec(
            vertices=[
                (-half_width - wall_t, half_width),
                (-half_width, half_width),
                (-half_width, top_y + wall_t),
                (-half_width - wall_t, top_y + wall_t),
            ]
        )

        spawn_half_width = max(1e-6, 0.5 * self.spawn_width_ratio * width)
        horizontal_spawn_depth = max(
            1e-6,
            min(self.spawn_depth_ratio * horizontal_length, horizontal_length),
        )
        vertical_spawn_depth = max(
            1e-6,
            min(self.spawn_depth_ratio * vertical_length, vertical_length),
        )

        left_spawn_region = RegionSpec(
            min_corner=(left_x, -spawn_half_width),
            max_corner=(left_x + horizontal_spawn_depth, spawn_half_width),
        )
        top_spawn_region = RegionSpec(
            min_corner=(-spawn_half_width, top_y - vertical_spawn_depth),
            max_corner=(spawn_half_width, top_y),
        )

        left_spawn_area = 2.0 * spawn_half_width * horizontal_spawn_depth
        top_spawn_area = 2.0 * spawn_half_width * vertical_spawn_depth
        left_spawn_count = max(0, math.floor(spawn_density * left_spawn_area))
        top_spawn_count = max(0, math.floor(spawn_density * top_spawn_area))

        turn_radius = max(
            1e-6,
            min(
                self.turn_radius_ratio * half_width,
                0.45 * horizontal_length,
                0.45 * vertical_length,
            ),
        )

        centerline_forward_points: List[Tuple[float, float]] = [(left_x, 0.0)]
        if left_x < -turn_radius:
            centerline_forward_points.append((-turn_radius, 0.0))

        arc_center_x = -turn_radius
        arc_center_y = turn_radius
        for idx in range(1, self.turn_smoothing_segments):
            angle = -0.5 * math.pi + (0.5 * math.pi * idx / self.turn_smoothing_segments)
            arc_x = arc_center_x + turn_radius * math.cos(angle)
            arc_y = arc_center_y + turn_radius * math.sin(angle)
            centerline_forward_points.append((float(arc_x), float(arc_y)))

        centerline_forward_points.append((0.0, turn_radius))
        if top_y > turn_radius:
            centerline_forward_points.append((0.0, top_y))

        centerline_forward = PathSpec(points=centerline_forward_points)
        centerline_reverse = PathSpec(points=list(reversed(centerline_forward_points)))
        paths = [centerline_forward]

        region_pairs = [
            RegionPairSpec(
                spawn_region=left_spawn_region,
                destination_region=top_spawn_region,
                startup_agent_count=left_spawn_count,
                path_index=0,
                velocity_range=self.spawn_velocity_range,
            )
        ]

        if self.num_region_pairs == 2:
            paths.append(centerline_reverse)
            region_pairs.append(
                RegionPairSpec(
                    spawn_region=top_spawn_region,
                    destination_region=left_spawn_region,
                    startup_agent_count=top_spawn_count,
                    path_index=1,
                    velocity_range=self.spawn_velocity_range,
                )
            )

        return Scene(
            agents=[],
            obstacles=[
                lower_wall,
                right_wall,
                lower_inner_wall,
                right_inner_wall,
            ],
            paths=paths,
            region_pairs=region_pairs,
        )
