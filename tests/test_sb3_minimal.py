from __future__ import annotations

import copy
import importlib

import numpy as np
import os
pytest = importlib.import_module("pytest")

pytest.importorskip("rvo2")
pytest.importorskip("gymnasium")

from src.scene import Scene
from src.templates import empty_goal_templates
from sb3.env_orca import ORCASB3Env, ORCASB3EnvConfig


def _single_empty_goal_scene() -> Scene:
    templates = empty_goal_templates(goal_distance_range=(2.0, 2.0), goal_seed=0)
    scenes = templates[0].generate()
    if not scenes:
        raise RuntimeError("empty_goal_templates did not generate scenes")
    return scenes[0]


def test_sb3_env_reset_and_step_shapes() -> None:
    base_scene = _single_empty_goal_scene()

    def scene_factory() -> Scene:
        return copy.deepcopy(base_scene)

    env = ORCASB3Env(scene_factory=scene_factory, config=ORCASB3EnvConfig(max_steps=8))

    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert "goal_distance" in info

    action = np.array([0.5, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in step_info


def test_sb3_ppo_wiring_runs_short_rollout() -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    ppo_cls = sb3.PPO

    from sb3.policy import MinimalActorCriticPolicy

    base_scene = _single_empty_goal_scene()

    def scene_factory() -> Scene:
        return copy.deepcopy(base_scene)

    env = ORCASB3Env(scene_factory=scene_factory, config=ORCASB3EnvConfig(max_steps=16))

    model = ppo_cls(
        policy=MinimalActorCriticPolicy,
        env=env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=0,
        verbose=0,
        device="cpu",
    )

    model.learn(total_timesteps=16)

    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert np.asarray(action).shape == (2,)

if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))
