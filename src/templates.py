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
    """Return premade scene templates used for rollouts.

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
        num_levels=15,
    )
    l_template = LShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.3, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
        num_levels=15,
    )
    t_template = TShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.2, 0.05),
        spawn_velocity_range=(0.8, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=3,
        num_levels=15,
    )
    cross_template = CrossShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.1, 0.02),
        spawn_velocity_range=(0.8, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=2,
        num_levels=4,
    )

    return [cross_template]


def test_templates():
    straight_template = StraightCorridorTemplate(
        width_range=(3.0, 10.0),
        length_range=(8.0, 20.0),
        spawn_density_range=(0.3, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
        num_levels=3,
    )
    l_template = LShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.3, 0.2),
        spawn_velocity_range=(0.8, 2.6),
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
        num_levels=3,
    )
    t_template = TShapeCorridorTemplate(
        width_range=(4.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(5.0, 15.0),
        spawn_density_range=(0.05, 0.05),
        spawn_velocity_range=(0.5, 3.0),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=1,
        num_levels=10,
    )
    cross_template = CrossShapeCorridorTemplate(
        width_range=(3.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.1, 0.02),
        spawn_velocity_range=(0.8, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=4,
        num_levels=3,
    )

    # return [straight_template, l_template, t_template, cross_template]
    return [t_template]


def cross_templates() -> SceneTemplate:
    cross = CrossShapeCorridorTemplate(
        width_range=(4.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.1, 0.05),
        spawn_velocity_range=(0.5, 2.6),
        spawn_depth_ratio=0.3,
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_enabled_start_regions=1,
        num_levels=20,
    )

    return [cross]