from __future__ import annotations

from typing import List

from src.scene_template import (
    EmptySingleAgentGoalTemplate,
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
        num_levels=5,
    )

    return [straight_template, l_template, t_template, cross_template]


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
        num_levels=10,
    )

    return [cross]

def l_shape_templates() -> SceneTemplate:
    l_shape = LShapeCorridorTemplate(
        width_range=(4.0, 10.0),
        horizontal_length_range=(8.0, 20.0),
        vertical_length_range=(8.0, 20.0),
        spawn_density_range=(0.1, 0.05),
        spawn_velocity_range=(0.5, 2.6),
        turn_radius_ratio=1.2,
        ego_center_spacing=2.0,
        ego_center_noise_std=0.1,
        num_region_pairs=2,
        num_levels=10,
    )

    return [l_shape]


def empty_goal_templates(
    goal_distance_range: tuple[float, float] = (2.0, 6.0),
    goal_seed: int | None = 0,
    num_other_agents_range: tuple[int, int] = (0, 0),
    other_agent_spawn_radius_range: tuple[float, float] = (1.5, 6.0),
    other_agent_goal_distance_range: tuple[float, float] = (2.0, 6.0),
    other_agent_min_start_separation: float = 0.8,
    num_levels: int = 32,
) -> list[SceneTemplate]:
    return [
        EmptySingleAgentGoalTemplate(
            goal_distance_range=goal_distance_range,
            goal_seed=goal_seed,
            num_other_agents=num_other_agents_range,
            other_agent_spawn_radius_range=other_agent_spawn_radius_range,
            other_agent_goal_distance_range=other_agent_goal_distance_range,
            other_agent_min_start_separation=other_agent_min_start_separation,
            num_levels=num_levels,
        )
    ]

def empty_goal_preset_templates() -> list[SceneTemplate]:
    return empty_goal_templates(
        goal_distance_range=(2.0, 8.0),
        goal_seed=42,
        num_other_agents_range=(2, 10),
        other_agent_spawn_radius_range=(2.0, 6.0),
        other_agent_goal_distance_range=(5.0, 15.0),
        other_agent_min_start_separation=1.0,
        num_levels=20,
    )