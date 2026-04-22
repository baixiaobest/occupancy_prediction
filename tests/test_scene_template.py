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


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))