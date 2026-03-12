from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AgentSpec:
    position: Tuple[float, float]
    goal: Tuple[float, float]


@dataclass
class ObstacleSpec:
    vertices: List[Tuple[float, float]]


@dataclass
class Scene:
    agents: List[AgentSpec]
    obstacles: List[ObstacleSpec]
