from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.scene import Scene
from src.scene_template import SceneTemplate
from src.templates import (
    cross_templates,
    default_templates,
    empty_goal_preset_templates,
    empty_goal_templates,
    l_shape_templates,
    test_templates,
)


@dataclass(frozen=True)
class EmptyGoalTemplateConfig:
    goal_distance_range: tuple[float, float] = (2.0, 6.0)
    goal_seed: int | None = 0
    num_other_agents_range: tuple[int, int] = (0, 0)
    other_agent_spawn_radius_range: tuple[float, float] = (1.5, 6.0)
    other_agent_goal_distance_range: tuple[float, float] = (2.0, 6.0)
    other_agent_min_start_separation: float = 0.8
    num_levels: int = 32


def seed_everything(seed: int, *, include_numpy: bool = True) -> None:
    random.seed(seed)
    if include_numpy:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_templates(
    template_set: str,
    *,
    empty_goal: EmptyGoalTemplateConfig | None = None,
    use_empty_goal_preset: bool = False,
) -> list[SceneTemplate]:
    normalized = str(template_set)
    if normalized == "default":
        return default_templates()
    if normalized == "test":
        return test_templates()
    if normalized == "cross":
        return cross_templates()
    if normalized == "l_shape":
        return l_shape_templates()
    if normalized == "empty_goal":
        if use_empty_goal_preset:
            return empty_goal_preset_templates()

        cfg = empty_goal or EmptyGoalTemplateConfig()
        return empty_goal_templates(
            goal_distance_range=tuple(float(v) for v in cfg.goal_distance_range),
            goal_seed=None if cfg.goal_seed is None else int(cfg.goal_seed),
            num_other_agents_range=tuple(int(v) for v in cfg.num_other_agents_range),
            other_agent_spawn_radius_range=tuple(float(v) for v in cfg.other_agent_spawn_radius_range),
            other_agent_goal_distance_range=tuple(float(v) for v in cfg.other_agent_goal_distance_range),
            other_agent_min_start_separation=float(cfg.other_agent_min_start_separation),
            num_levels=int(cfg.num_levels),
        )
    raise ValueError(f"Unknown template set: {template_set}")


def build_scene_pool(
    template_set: str,
    *,
    empty_goal: EmptyGoalTemplateConfig | None = None,
    use_empty_goal_preset: bool = False,
    scene_filter: Callable[[Scene], bool] | None = None,
) -> list[Scene]:
    templates = select_templates(
        template_set,
        empty_goal=empty_goal,
        use_empty_goal_preset=use_empty_goal_preset,
    )

    scenes: list[Scene] = []
    for template in templates:
        scenes.extend(template.generate())

    if scene_filter is not None:
        scenes = [scene for scene in scenes if scene_filter(scene)]

    if not scenes:
        raise ValueError(f"No scenes generated for template set: {template_set}")

    return scenes