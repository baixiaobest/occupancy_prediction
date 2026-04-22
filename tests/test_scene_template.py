from __future__ import annotations

import math
import os

import pytest

from src.scene_template import EmptySingleAgentGoalTemplate


def test_empty_single_agent_goal_template_generates_expected_scenes() -> None:
    template = EmptySingleAgentGoalTemplate(
        goal_distance_range=(2.0, 4.0),
        num_levels=3,
        start_position=(0.0, 0.0),
        goal_seed=7,
    )

    scenes = template.generate()

    assert template.get_name() == "empty_single_agent_goal"
    assert len(scenes) == 3

    expected_radii = [2.0, 3.0, 4.0]
    for scene, expected_radius in zip(scenes, expected_radii):
        assert len(scene.agents) == 1
        assert scene.obstacles == []
        assert scene.paths == []
        assert scene.region_pairs == []
        assert scene.ego_centers == [(0.0, 0.0)]

        agent = scene.agents[0]
        assert agent.position == (0.0, 0.0)
        assert agent.path_index is None

        goal_radius = math.hypot(agent.goal[0], agent.goal[1])
        assert goal_radius == pytest.approx(expected_radius)


def test_empty_single_agent_goal_template_generates_other_agents() -> None:
    template = EmptySingleAgentGoalTemplate(
        goal_distance_range=(3.0, 3.0),
        num_levels=2,
        start_position=(0.0, 0.0),
        goal_seed=11,
        num_other_agents=(3, 3),
        other_agent_spawn_radius_range=(2.0, 2.0),
        other_agent_goal_distance_range=(1.5, 1.5),
        other_agent_min_start_separation=0.5,
    )

    scenes = template.generate()

    assert len(scenes) == 2
    for scene in scenes:
        # Ego is always the first agent.
        assert len(scene.agents) == 4
        ego = scene.agents[0]
        assert ego.position == (0.0, 0.0)

        for agent in scene.agents[1:]:
            start = agent.position
            goal = agent.goal
            start_radius = math.hypot(start[0], start[1])
            assert start_radius == pytest.approx(2.0)

            goal_dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
            assert goal_dist == pytest.approx(1.5)


def test_empty_single_agent_goal_template_other_agent_range_is_respected() -> None:
    template = EmptySingleAgentGoalTemplate(
        goal_distance_range=(2.0, 2.0),
        num_levels=8,
        goal_seed=19,
        num_other_agents=(1, 3),
        other_agent_spawn_radius_range=(1.0, 1.0),
        other_agent_goal_distance_range=(1.0, 1.0),
        other_agent_min_start_separation=0.1,
    )

    scenes = template.generate()
    counts = [len(scene.agents) - 1 for scene in scenes]
    assert len(scenes) == 8
    assert all(1 <= count <= 3 for count in counts)


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))