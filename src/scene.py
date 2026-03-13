from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RegionSpec:
    """Axis-aligned rectangular region represented by min/max corners."""

    min_corner: Tuple[float, float]
    max_corner: Tuple[float, float]


@dataclass
class PathSpec:
    """Polyline reference path represented by ordered 2D waypoints."""

    points: List[Tuple[float, float]]


@dataclass
class AgentSpec:
    """Agent configuration with initial position, goal, and optional path binding."""

    position: Tuple[float, float]
    goal: Tuple[float, float]
    path_index: int | None = None


@dataclass
class ObstacleSpec:
    """Polygon obstacle specified by ordered vertices."""

    vertices: List[Tuple[float, float]]


@dataclass
class RegionPairSpec:
    """Spawn-destination region pair with startup-only spawning behavior."""

    spawn_region: RegionSpec
    destination_region: RegionSpec
    startup_agent_count: int = 0
    path_index: int | None = None
    velocity_range: Tuple[float, float] | None = None


@dataclass
class Scene:
    """Complete simulation scene containing agents, obstacles, and optional paths."""

    agents: List[AgentSpec]
    obstacles: List[ObstacleSpec]
    paths: List[PathSpec] = field(default_factory=list)
    region_pairs: List[RegionPairSpec] = field(default_factory=list)
    ego_centers: List[Tuple[float, float]] = field(default_factory=list)
