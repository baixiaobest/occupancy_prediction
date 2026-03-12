from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PathSpec:
    points: List[Tuple[float, float]]


@dataclass
class AgentSpec:
    position: Tuple[float, float]
    goal: Tuple[float, float]
    path_index: int | None = None


@dataclass
class ObstacleSpec:
    vertices: List[Tuple[float, float]]


@dataclass
class Scene:
    agents: List[AgentSpec]
    obstacles: List[ObstacleSpec]
    paths: List[PathSpec] = field(default_factory=list)
