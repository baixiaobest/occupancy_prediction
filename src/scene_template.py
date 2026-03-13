from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import List, Tuple

import numpy as np

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

    @staticmethod
    def _spawn_count_from_density(spawn_density: float, spawn_area: float) -> int:
        """Compute startup spawn count from density and area.

        Uses floor(density * area), but enforces at least one agent when
        `spawn_density > 0`.
        """
        if spawn_density <= 0.0:
            return 0
        return max(1, math.floor(spawn_density * spawn_area))

    @staticmethod
    def _compute_turn_radius(
        half_width: float,
        primary_length: float,
        secondary_length: float,
        turn_radius_ratio: float,
    ) -> float:
        """Compute a bounded corner-smoothing radius for corridor turns."""
        return max(
            1e-6,
            min(
                turn_radius_ratio * half_width,
                0.45 * primary_length,
                0.45 * secondary_length,
            ),
        )

    @staticmethod
    def _arc_interior_points(
        center_x: float,
        center_y: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        num_segments: int,
    ) -> List[Tuple[float, float]]:
        """Generate interior waypoints on an arc, excluding endpoints."""
        if num_segments <= 1:
            return []

        points: List[Tuple[float, float]] = []
        for idx in range(1, num_segments):
            alpha = idx / num_segments
            angle = start_angle + (end_angle - start_angle) * alpha
            points.append(
                (
                    float(center_x + radius * math.cos(angle)),
                    float(center_y + radius * math.sin(angle)),
                )
            )
        return points

    @staticmethod
    def _sample_points_on_polyline(
        polyline: List[Tuple[float, float]], spacing: float
    ) -> List[Tuple[float, float]]:
        """Sample points along a polyline with approximately uniform spacing."""
        if not polyline:
            return []
        if len(polyline) == 1:
            return [polyline[0]]
        if spacing <= 0.0:
            raise ValueError("spacing must be > 0")

        sampled: List[Tuple[float, float]] = [polyline[0]]
        next_target = spacing
        traversed = 0.0

        for idx in range(len(polyline) - 1):
            x0, y0 = polyline[idx]
            x1, y1 = polyline[idx + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-9:
                continue

            while next_target <= traversed + seg_len + 1e-9:
                alpha = (next_target - traversed) / seg_len
                sampled.append((float(x0 + alpha * dx), float(y0 + alpha * dy)))
                next_target += spacing

            traversed += seg_len

        end_point = polyline[-1]
        if math.hypot(sampled[-1][0] - end_point[0], sampled[-1][1] - end_point[1]) > 1e-6:
            sampled.append(end_point)

        return sampled

    @staticmethod
    def _merge_unique_points(
        points: List[Tuple[float, float]], precision: int = 6
    ) -> List[Tuple[float, float]]:
        """Merge duplicate points while preserving insertion order."""
        merged: List[Tuple[float, float]] = []
        seen: set[Tuple[float, float]] = set()
        for x, y in points:
            key = (round(float(x), precision), round(float(y), precision))
            if key in seen:
                continue
            seen.add(key)
            merged.append((float(x), float(y)))
        return merged

    @staticmethod
    def _jitter_points(
        points: List[Tuple[float, float]],
        noise_std: float,
        rng: np.random.Generator,
    ) -> List[Tuple[float, float]]:
        """Apply Gaussian XY jitter to points."""
        if noise_std <= 0.0:
            return list(points)

        jittered: List[Tuple[float, float]] = []
        for x, y in points:
            noise = rng.normal(loc=0.0, scale=noise_std, size=2)
            jittered.append((float(x + noise[0]), float(y + noise[1])))
        return jittered


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
        ego_center_spacing: float = 1.0,
        ego_center_noise_std: float = 0.1,
        ego_center_noise_seed: int | None = 0,
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
        if ego_center_spacing <= 0.0:
            raise ValueError("ego_center_spacing must be > 0")
        if ego_center_noise_std < 0.0:
            raise ValueError("ego_center_noise_std must be >= 0")

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
        self.ego_center_spacing = float(ego_center_spacing)
        self.ego_center_noise_std = float(ego_center_noise_std)
        self._ego_center_rng = np.random.default_rng(ego_center_noise_seed)

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
        startup_agent_count = self._spawn_count_from_density(spawn_density, spawn_area)

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

        ego_center_base = self._sample_points_on_polyline(
            [(0.0, 0.0), (length, 0.0)],
            spacing=self.ego_center_spacing,
        )
        ego_centers = self._jitter_points(
            ego_center_base,
            noise_std=self.ego_center_noise_std,
            rng=self._ego_center_rng,
        )

        return Scene(
            agents=[],
            obstacles=[bottom_wall, top_wall],
            paths=[],
            region_pairs=region_pairs,
            ego_centers=ego_centers,
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
        ego_center_spacing: float = 1.0,
        ego_center_noise_std: float = 0.05,
        ego_center_noise_seed: int | None = 0,
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
        if ego_center_spacing <= 0.0:
            raise ValueError("ego_center_spacing must be > 0")
        if ego_center_noise_std < 0.0:
            raise ValueError("ego_center_noise_std must be >= 0")

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
        self.ego_center_spacing = float(ego_center_spacing)
        self.ego_center_noise_std = float(ego_center_noise_std)
        self._ego_center_rng = np.random.default_rng(ego_center_noise_seed)

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
        left_spawn_count = self._spawn_count_from_density(spawn_density, left_spawn_area)
        top_spawn_count = self._spawn_count_from_density(spawn_density, top_spawn_area)

        turn_radius = self._compute_turn_radius(
            half_width=half_width,
            primary_length=horizontal_length,
            secondary_length=vertical_length,
            turn_radius_ratio=self.turn_radius_ratio,
        )

        centerline_forward_points: List[Tuple[float, float]] = [(left_x, 0.0)]
        if left_x < -turn_radius:
            centerline_forward_points.append((-turn_radius, 0.0))

        centerline_forward_points.extend(
            self._arc_interior_points(
                center_x=-turn_radius,
                center_y=turn_radius,
                radius=turn_radius,
                start_angle=-0.5 * math.pi,
                end_angle=0.0,
                num_segments=self.turn_smoothing_segments,
            )
        )

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

        ego_center_base = self._sample_points_on_polyline(
            centerline_forward_points,
            spacing=self.ego_center_spacing,
        )
        ego_centers = self._jitter_points(
            ego_center_base,
            noise_std=self.ego_center_noise_std,
            rng=self._ego_center_rng,
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
            ego_centers=ego_centers,
        )


class TShapeCorridorTemplate(SceneTemplate):
    """Template for T-shaped corridor scenes with grouped start terminals.

    Terminal centers are:
    - left: (-horizontal_length, 0)
    - right: (horizontal_length, 0)
    - bottom: (0, -vertical_length)

    For all three starts enabled, this yields 6 directed region pairs (each start has
    two possible destinations).
    """

    def __init__(
        self,
        width_range: Tuple[float, float],
        horizontal_length_range: Tuple[float, float],
        vertical_length_range: Tuple[float, float],
        spawn_density_range: Tuple[float, float] = (1.0, 1.0),
        spawn_velocity_range: Tuple[float, float] = (1.5, 1.5),
        num_enabled_start_regions: int = 3,
        spawn_depth_ratio: float = 0.15,
        spawn_width_ratio: float = 0.8,
        wall_thickness: float = 0.4,
        turn_smoothing_segments: int = 6,
        turn_radius_ratio: float = 0.6,
        ego_center_spacing: float = 1.0,
        ego_center_noise_std: float = 0.05,
        ego_center_noise_seed: int | None = 0,
    ) -> None:
        if num_enabled_start_regions not in (1, 2, 3):
            raise ValueError("num_enabled_start_regions must be 1, 2, or 3")
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
        if ego_center_spacing <= 0.0:
            raise ValueError("ego_center_spacing must be > 0")
        if ego_center_noise_std < 0.0:
            raise ValueError("ego_center_noise_std must be >= 0")

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
        self.num_enabled_start_regions = int(num_enabled_start_regions)
        self.spawn_depth_ratio = float(spawn_depth_ratio)
        self.spawn_width_ratio = float(spawn_width_ratio)
        self.wall_thickness = float(wall_thickness)
        self.turn_smoothing_segments = int(turn_smoothing_segments)
        self.turn_radius_ratio = float(turn_radius_ratio)
        self.ego_center_spacing = float(ego_center_spacing)
        self.ego_center_noise_std = float(ego_center_noise_std)
        self._ego_center_rng = np.random.default_rng(ego_center_noise_seed)

    def generate(self, num_levels: int) -> List[Scene]:
        """Generate `num_levels` T-shaped corridor scenes."""
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
        right_x = horizontal_length
        bottom_y = -vertical_length

        upper_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, half_width),
                (right_x + wall_t, half_width),
                (right_x + wall_t, half_width + wall_t),
                (left_x - wall_t, half_width + wall_t),
            ]
        )
        lower_left_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, -half_width - wall_t),
                (-half_width, -half_width - wall_t),
                (-half_width, -half_width),
                (left_x - wall_t, -half_width),
            ]
        )
        lower_right_wall = ObstacleSpec(
            vertices=[
                (half_width, -half_width - wall_t),
                (right_x + wall_t, -half_width - wall_t),
                (right_x + wall_t, -half_width),
                (half_width, -half_width),
            ]
        )
        stem_left_wall = ObstacleSpec(
            vertices=[
                (-half_width - wall_t, bottom_y - wall_t),
                (-half_width, bottom_y - wall_t),
                (-half_width, -half_width),
                (-half_width - wall_t, -half_width),
            ]
        )
        stem_right_wall = ObstacleSpec(
            vertices=[
                (half_width, bottom_y - wall_t),
                (half_width + wall_t, bottom_y - wall_t),
                (half_width + wall_t, -half_width),
                (half_width, -half_width),
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

        left_region = RegionSpec(
            min_corner=(left_x, -spawn_half_width),
            max_corner=(left_x + horizontal_spawn_depth, spawn_half_width),
        )
        right_region = RegionSpec(
            min_corner=(right_x - horizontal_spawn_depth, -spawn_half_width),
            max_corner=(right_x, spawn_half_width),
        )
        bottom_region = RegionSpec(
            min_corner=(-spawn_half_width, bottom_y),
            max_corner=(spawn_half_width, bottom_y + vertical_spawn_depth),
        )

        left_spawn_area = 2.0 * spawn_half_width * horizontal_spawn_depth
        right_spawn_area = 2.0 * spawn_half_width * horizontal_spawn_depth
        bottom_spawn_area = 2.0 * spawn_half_width * vertical_spawn_depth
        left_spawn_count = self._spawn_count_from_density(spawn_density, left_spawn_area)
        right_spawn_count = self._spawn_count_from_density(spawn_density, right_spawn_area)
        bottom_spawn_count = self._spawn_count_from_density(spawn_density, bottom_spawn_area)

        terminal_points = {
            "left": (left_x, 0.0),
            "right": (right_x, 0.0),
            "bottom": (0.0, bottom_y),
        }
        terminal_regions = {
            "left": left_region,
            "right": right_region,
            "bottom": bottom_region,
        }
        terminal_counts = {
            "left": left_spawn_count,
            "right": right_spawn_count,
            "bottom": bottom_spawn_count,
        }

        all_pairs = [
            ("left", "right"),
            ("left", "bottom"),
            ("right", "left"),
            ("right", "bottom"),
            ("bottom", "left"),
            ("bottom", "right"),
        ]

        enabled_starts = ["left", "right", "bottom"][: self.num_enabled_start_regions]
        selected_pairs = [pair for pair in all_pairs if pair[0] in enabled_starts]

        turn_radius = self._compute_turn_radius(
            half_width=half_width,
            primary_length=horizontal_length,
            secondary_length=vertical_length,
            turn_radius_ratio=self.turn_radius_ratio,
        )

        def _build_turn_path(start_name: str, end_name: str) -> List[Tuple[float, float]]:
            if start_name == "left" and end_name == "right":
                return [terminal_points["left"], terminal_points["right"]]
            if start_name == "right" and end_name == "left":
                return [terminal_points["right"], terminal_points["left"]]

            points: List[Tuple[float, float]] = []

            if start_name == "left" and end_name == "bottom":
                points.append((left_x, 0.0))
                if left_x < -turn_radius:
                    points.append((-turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.5 * math.pi,
                        end_angle=0.0,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, -turn_radius))
                if bottom_y < -turn_radius:
                    points.append((0.0, bottom_y))
                return points

            if start_name == "right" and end_name == "bottom":
                points.append((right_x, 0.0))
                if right_x > turn_radius:
                    points.append((turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.5 * math.pi,
                        end_angle=math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, -turn_radius))
                if bottom_y < -turn_radius:
                    points.append((0.0, bottom_y))
                return points

            if start_name == "bottom" and end_name == "left":
                points.append((0.0, bottom_y))
                if bottom_y < -turn_radius:
                    points.append((0.0, -turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.0,
                        end_angle=0.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((-turn_radius, 0.0))
                if left_x < -turn_radius:
                    points.append((left_x, 0.0))
                return points

            if start_name == "bottom" and end_name == "right":
                points.append((0.0, bottom_y))
                if bottom_y < -turn_radius:
                    points.append((0.0, -turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=math.pi,
                        end_angle=0.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((turn_radius, 0.0))
                if right_x > turn_radius:
                    points.append((right_x, 0.0))
                return points

            return [terminal_points[start_name], (0.0, 0.0), terminal_points[end_name]]

        paths: List[PathSpec] = []
        region_pairs: List[RegionPairSpec] = []
        for start_name, end_name in selected_pairs:
            path_points = _build_turn_path(start_name, end_name)

            path_index = len(paths)
            paths.append(PathSpec(points=path_points))
            region_pairs.append(
                RegionPairSpec(
                    spawn_region=terminal_regions[start_name],
                    destination_region=terminal_regions[end_name],
                    startup_agent_count=terminal_counts[start_name],
                    path_index=path_index,
                    velocity_range=self.spawn_velocity_range,
                )
            )

        horizontal_centers = self._sample_points_on_polyline(
            [(left_x, 0.0), (right_x, 0.0)],
            spacing=self.ego_center_spacing,
        )
        vertical_centers = self._sample_points_on_polyline(
            [(0.0, bottom_y), (0.0, 0.0)],
            spacing=self.ego_center_spacing,
        )
        ego_center_base = self._merge_unique_points(horizontal_centers + vertical_centers)
        ego_centers = self._jitter_points(
            ego_center_base,
            noise_std=self.ego_center_noise_std,
            rng=self._ego_center_rng,
        )

        return Scene(
            agents=[],
            obstacles=[
                upper_wall,
                lower_left_wall,
                lower_right_wall,
                stem_left_wall,
                stem_right_wall,
            ],
            paths=paths,
            region_pairs=region_pairs,
            ego_centers=ego_centers,
        )


class CrossShapeCorridorTemplate(SceneTemplate):
    """Template for cross-shaped corridor scenes with grouped start terminals.

    Terminal centers are:
    - left: (-horizontal_length, 0)
    - right: (horizontal_length, 0)
    - bottom: (0, -vertical_length)
    - top: (0, vertical_length)

    With all starts enabled, this yields 12 directed region pairs
    (4 starts x 3 destinations each).
    """

    def __init__(
        self,
        width_range: Tuple[float, float],
        horizontal_length_range: Tuple[float, float],
        vertical_length_range: Tuple[float, float],
        spawn_density_range: Tuple[float, float] = (1.0, 1.0),
        spawn_velocity_range: Tuple[float, float] = (1.5, 1.5),
        num_enabled_start_regions: int = 4,
        spawn_depth_ratio: float = 0.15,
        spawn_width_ratio: float = 0.8,
        wall_thickness: float = 0.4,
        turn_smoothing_segments: int = 6,
        turn_radius_ratio: float = 0.6,
        ego_center_spacing: float = 1.0,
        ego_center_noise_std: float = 0.05,
        ego_center_noise_seed: int | None = 0,
    ) -> None:
        if num_enabled_start_regions not in (1, 2, 3, 4):
            raise ValueError("num_enabled_start_regions must be 1, 2, 3, or 4")
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
        if ego_center_spacing <= 0.0:
            raise ValueError("ego_center_spacing must be > 0")
        if ego_center_noise_std < 0.0:
            raise ValueError("ego_center_noise_std must be >= 0")

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
        self.num_enabled_start_regions = int(num_enabled_start_regions)
        self.spawn_depth_ratio = float(spawn_depth_ratio)
        self.spawn_width_ratio = float(spawn_width_ratio)
        self.wall_thickness = float(wall_thickness)
        self.turn_smoothing_segments = int(turn_smoothing_segments)
        self.turn_radius_ratio = float(turn_radius_ratio)
        self.ego_center_spacing = float(ego_center_spacing)
        self.ego_center_noise_std = float(ego_center_noise_std)
        self._ego_center_rng = np.random.default_rng(ego_center_noise_seed)

    def generate(self, num_levels: int) -> List[Scene]:
        """Generate `num_levels` cross-shaped corridor scenes."""
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
        right_x = horizontal_length
        bottom_y = -vertical_length
        top_y = vertical_length

        upper_left_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, half_width),
                (-half_width, half_width),
                (-half_width, half_width + wall_t),
                (left_x - wall_t, half_width + wall_t),
            ]
        )
        upper_right_wall = ObstacleSpec(
            vertices=[
                (half_width, half_width),
                (right_x + wall_t, half_width),
                (right_x + wall_t, half_width + wall_t),
                (half_width, half_width + wall_t),
            ]
        )
        lower_left_wall = ObstacleSpec(
            vertices=[
                (left_x - wall_t, -half_width - wall_t),
                (-half_width, -half_width - wall_t),
                (-half_width, -half_width),
                (left_x - wall_t, -half_width),
            ]
        )
        lower_right_wall = ObstacleSpec(
            vertices=[
                (half_width, -half_width - wall_t),
                (right_x + wall_t, -half_width - wall_t),
                (right_x + wall_t, -half_width),
                (half_width, -half_width),
            ]
        )
        top_stem_left_wall = ObstacleSpec(
            vertices=[
                (-half_width - wall_t, half_width),
                (-half_width, half_width),
                (-half_width, top_y + wall_t),
                (-half_width - wall_t, top_y + wall_t),
            ]
        )
        top_stem_right_wall = ObstacleSpec(
            vertices=[
                (half_width, half_width),
                (half_width + wall_t, half_width),
                (half_width + wall_t, top_y + wall_t),
                (half_width, top_y + wall_t),
            ]
        )
        bottom_stem_left_wall = ObstacleSpec(
            vertices=[
                (-half_width - wall_t, bottom_y - wall_t),
                (-half_width, bottom_y - wall_t),
                (-half_width, -half_width),
                (-half_width - wall_t, -half_width),
            ]
        )
        bottom_stem_right_wall = ObstacleSpec(
            vertices=[
                (half_width, bottom_y - wall_t),
                (half_width + wall_t, bottom_y - wall_t),
                (half_width + wall_t, -half_width),
                (half_width, -half_width),
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

        left_region = RegionSpec(
            min_corner=(left_x, -spawn_half_width),
            max_corner=(left_x + horizontal_spawn_depth, spawn_half_width),
        )
        right_region = RegionSpec(
            min_corner=(right_x - horizontal_spawn_depth, -spawn_half_width),
            max_corner=(right_x, spawn_half_width),
        )
        bottom_region = RegionSpec(
            min_corner=(-spawn_half_width, bottom_y),
            max_corner=(spawn_half_width, bottom_y + vertical_spawn_depth),
        )
        top_region = RegionSpec(
            min_corner=(-spawn_half_width, top_y - vertical_spawn_depth),
            max_corner=(spawn_half_width, top_y),
        )

        horizontal_spawn_area = 2.0 * spawn_half_width * horizontal_spawn_depth
        vertical_spawn_area = 2.0 * spawn_half_width * vertical_spawn_depth
        left_spawn_count = self._spawn_count_from_density(spawn_density, horizontal_spawn_area)
        right_spawn_count = self._spawn_count_from_density(spawn_density, horizontal_spawn_area)
        bottom_spawn_count = self._spawn_count_from_density(spawn_density, vertical_spawn_area)
        top_spawn_count = self._spawn_count_from_density(spawn_density, vertical_spawn_area)

        terminal_points = {
            "left": (left_x, 0.0),
            "right": (right_x, 0.0),
            "bottom": (0.0, bottom_y),
            "top": (0.0, top_y),
        }
        terminal_regions = {
            "left": left_region,
            "right": right_region,
            "bottom": bottom_region,
            "top": top_region,
        }
        terminal_counts = {
            "left": left_spawn_count,
            "right": right_spawn_count,
            "bottom": bottom_spawn_count,
            "top": top_spawn_count,
        }

        starts_in_order = ["left", "right", "bottom", "top"]
        enabled_starts = starts_in_order[: self.num_enabled_start_regions]
        all_pairs = [
            (start_name, end_name)
            for start_name in starts_in_order
            for end_name in starts_in_order
            if start_name != end_name
        ]
        selected_pairs = [pair for pair in all_pairs if pair[0] in enabled_starts]

        turn_radius = self._compute_turn_radius(
            half_width=half_width,
            primary_length=horizontal_length,
            secondary_length=vertical_length,
            turn_radius_ratio=self.turn_radius_ratio,
        )

        def _build_turn_path(start_name: str, end_name: str) -> List[Tuple[float, float]]:
            if (start_name, end_name) in (("left", "right"), ("right", "left")):
                return [terminal_points[start_name], terminal_points[end_name]]
            if (start_name, end_name) in (("top", "bottom"), ("bottom", "top")):
                return [terminal_points[start_name], terminal_points[end_name]]

            points: List[Tuple[float, float]] = []

            if start_name == "left" and end_name == "top":
                points.append((left_x, 0.0))
                if left_x < -turn_radius:
                    points.append((-turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=turn_radius,
                        radius=turn_radius,
                        start_angle=-0.5 * math.pi,
                        end_angle=0.0,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, turn_radius))
                if top_y > turn_radius:
                    points.append((0.0, top_y))
                return points

            if start_name == "left" and end_name == "bottom":
                points.append((left_x, 0.0))
                if left_x < -turn_radius:
                    points.append((-turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.5 * math.pi,
                        end_angle=0.0,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, -turn_radius))
                if bottom_y < -turn_radius:
                    points.append((0.0, bottom_y))
                return points

            if start_name == "right" and end_name == "top":
                points.append((right_x, 0.0))
                if right_x > turn_radius:
                    points.append((turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=turn_radius,
                        radius=turn_radius,
                        start_angle=-0.5 * math.pi,
                        end_angle=-math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, turn_radius))
                if top_y > turn_radius:
                    points.append((0.0, top_y))
                return points

            if start_name == "right" and end_name == "bottom":
                points.append((right_x, 0.0))
                if right_x > turn_radius:
                    points.append((turn_radius, 0.0))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.5 * math.pi,
                        end_angle=math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((0.0, -turn_radius))
                if bottom_y < -turn_radius:
                    points.append((0.0, bottom_y))
                return points

            if start_name == "bottom" and end_name == "left":
                points.append((0.0, bottom_y))
                if bottom_y < -turn_radius:
                    points.append((0.0, -turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=0.0,
                        end_angle=0.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((-turn_radius, 0.0))
                if left_x < -turn_radius:
                    points.append((left_x, 0.0))
                return points

            if start_name == "bottom" and end_name == "right":
                points.append((0.0, bottom_y))
                if bottom_y < -turn_radius:
                    points.append((0.0, -turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=-turn_radius,
                        radius=turn_radius,
                        start_angle=math.pi,
                        end_angle=0.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((turn_radius, 0.0))
                if right_x > turn_radius:
                    points.append((right_x, 0.0))
                return points

            if start_name == "top" and end_name == "left":
                points.append((0.0, top_y))
                if top_y > turn_radius:
                    points.append((0.0, turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=-turn_radius,
                        center_y=turn_radius,
                        radius=turn_radius,
                        start_angle=0.0,
                        end_angle=-0.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((-turn_radius, 0.0))
                if left_x < -turn_radius:
                    points.append((left_x, 0.0))
                return points

            if start_name == "top" and end_name == "right":
                points.append((0.0, top_y))
                if top_y > turn_radius:
                    points.append((0.0, turn_radius))
                points.extend(
                    self._arc_interior_points(
                        center_x=turn_radius,
                        center_y=turn_radius,
                        radius=turn_radius,
                        start_angle=math.pi,
                        end_angle=1.5 * math.pi,
                        num_segments=self.turn_smoothing_segments,
                    )
                )
                points.append((turn_radius, 0.0))
                if right_x > turn_radius:
                    points.append((right_x, 0.0))
                return points

            return [terminal_points[start_name], (0.0, 0.0), terminal_points[end_name]]

        paths: List[PathSpec] = []
        region_pairs: List[RegionPairSpec] = []
        for start_name, end_name in selected_pairs:
            path_points = _build_turn_path(start_name, end_name)

            path_index = len(paths)
            paths.append(PathSpec(points=path_points))
            region_pairs.append(
                RegionPairSpec(
                    spawn_region=terminal_regions[start_name],
                    destination_region=terminal_regions[end_name],
                    startup_agent_count=terminal_counts[start_name],
                    path_index=path_index,
                    velocity_range=self.spawn_velocity_range,
                )
            )

        horizontal_centers = self._sample_points_on_polyline(
            [(left_x, 0.0), (right_x, 0.0)],
            spacing=self.ego_center_spacing,
        )
        vertical_centers = self._sample_points_on_polyline(
            [(0.0, bottom_y), (0.0, top_y)],
            spacing=self.ego_center_spacing,
        )
        ego_center_base = self._merge_unique_points(horizontal_centers + vertical_centers)
        ego_centers = self._jitter_points(
            ego_center_base,
            noise_std=self.ego_center_noise_std,
            rng=self._ego_center_rng,
        )

        return Scene(
            agents=[],
            obstacles=[
                upper_left_wall,
                upper_right_wall,
                lower_left_wall,
                lower_right_wall,
                top_stem_left_wall,
                top_stem_right_wall,
                bottom_stem_left_wall,
                bottom_stem_right_wall,
            ],
            paths=paths,
            region_pairs=region_pairs,
            ego_centers=ego_centers,
        )
