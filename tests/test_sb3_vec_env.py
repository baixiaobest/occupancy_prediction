from __future__ import annotations

import copy
import importlib
import os

import numpy as np

pytest = importlib.import_module("pytest")

pytest.importorskip("rvo2")
pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from src.scene import Scene
from src.templates import empty_goal_templates
from sb3.env_orca import ORCASB3EnvConfig
from sb3.vec_env_orca import build_orca_vec_env


def _single_empty_goal_scene() -> Scene:
    templates = empty_goal_templates(goal_distance_range=(2.0, 2.0), goal_seed=0)
    scenes = templates[0].generate()
    if not scenes:
        raise RuntimeError("empty_goal_templates did not generate scenes")
    return scenes[0]


def test_orca_vec_env_dummy_dict_observation_shapes() -> None:
    base_scene = _single_empty_goal_scene()
    scenes = [copy.deepcopy(base_scene) for _ in range(4)]

    vec_env = build_orca_vec_env(
        scenes=scenes,
        selection="cycle",
        fixed_scene_index=0,
        seed=123,
        num_envs=2,
        env_config=ORCASB3EnvConfig(max_steps=8),
        backend="dummy",
        minimal_observation=False,
    )

    try:
        obs = vec_env.reset()
        assert isinstance(obs, dict)
        assert obs["dynamic_context"].shape[0] == 2
        assert obs["static_map"].shape[0] == 2
        assert obs["goal_position"].shape == (2, 2)
        assert obs["current_velocity"].shape == (2, 2)
        assert obs["last_commanded_velocity"].shape == (2, 2)

        actions = np.zeros((2, 2), dtype=np.float32)
        next_obs, rewards, dones, infos = vec_env.step(actions)

        assert isinstance(next_obs, dict)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        vec_env.close()


def test_orca_vec_env_dummy_minimal_observation_shapes() -> None:
    base_scene = _single_empty_goal_scene()
    scenes = [copy.deepcopy(base_scene) for _ in range(2)]

    vec_env = build_orca_vec_env(
        scenes=scenes,
        selection="random",
        fixed_scene_index=0,
        seed=99,
        num_envs=2,
        env_config=ORCASB3EnvConfig(max_steps=8),
        backend="dummy",
        minimal_observation=True,
    )

    try:
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, 6)

        actions = np.zeros((2, 2), dtype=np.float32)
        next_obs, rewards, dones, infos = vec_env.step(actions)

        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (2, 6)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        vec_env.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
