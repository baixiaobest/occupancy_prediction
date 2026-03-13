from __future__ import annotations

from typing import List

from src.scene_template import (
    SceneTemplate,
    StraightCorridorTemplate,
    LShapeCorridorTemplate,
    TShapeCorridorTemplate,
    CrossShapeCorridorTemplate,
)


def default_templates() -> List[SceneTemplate]:
    """Return a list of premade scene-template instances used for rollouts.

    These mirror the templates previously embedded in the rollout script.
    """
    straight_template = StraightCorridorTemplate(
        width_range=(3.0, 10.0),
        length_range=(8.0, 20.0),
        spawn_density_range=(0.3, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
    )
    l_template = LShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.5, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
    )
    t_template = TShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.3, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=3,
    )
    cross_template = CrossShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.1, 0.1),
        spawn_velocity_range=(0.8, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=4,
    )

    return [straight_template, l_template, t_template, cross_template]
