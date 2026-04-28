from __future__ import annotations

import importlib
from pathlib import Path

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")
pytest.importorskip("skrl")
pytest.importorskip("rvo2")

import gymnasium as gym
import numpy as np
import torch

from src.scene import AgentSpec, Scene
from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig
from src.skrl import pipeline


def _single_agent_scene() -> Scene:
    return Scene(agents=[AgentSpec(position=(0.0, 0.0), goal=(1.0, 0.0))], obstacles=[])


def test_dump_effective_configs_stringifies_output() -> None:
    env_cfg = SkrlEnvBuildConfig()
    train_cfg = SkrlPPOTrainConfig(output=Path("checkpoints/x.pt"))
    dumped = pipeline.dump_effective_configs(env_cfg, train_cfg)
    assert dumped["train"]["output"] == "checkpoints/x.pt"


def test_make_single_env_observation_mode_switch(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_build_scene_pool", lambda config, seed: [_single_agent_scene()])
    monkeypatch.setattr(pipeline, "make_scene_factory", lambda scenes, selection, fixed_scene_index, seed: lambda: scenes[0])

    class _DummyBaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, scene_factory, config):
            super().__init__()
            self.scene_factory = scene_factory
            self.config = config
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros((2,), dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros((2,), dtype=np.float32), 0.0, False, False, {}

    class _DummyMinimalWrapper(gym.Wrapper):
        pass

    monkeypatch.setattr(pipeline, "TorchORCAEnv", _DummyBaseEnv)
    monkeypatch.setattr(pipeline, "MinimalKinematicsObservationWrapper", _DummyMinimalWrapper)

    env_cfg = SkrlEnvBuildConfig(observation_mode="occupancy")
    env_occ = pipeline._make_single_env(env_cfg, seed=0, device=torch.device("cpu"))
    assert isinstance(env_occ, _DummyBaseEnv)

    env_cfg_min = SkrlEnvBuildConfig(observation_mode="minimal")
    env_min = pipeline._make_single_env(env_cfg_min, seed=0, device=torch.device("cpu"))
    assert isinstance(env_min, _DummyMinimalWrapper)

    with pytest.raises(ValueError, match="Unknown observation_mode"):
        pipeline._make_single_env(
            SkrlEnvBuildConfig(observation_mode="bad_mode"),
            seed=0,
            device=torch.device("cpu"),
        )


def test_make_single_env_requires_vae_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_build_scene_pool", lambda config, seed: [_single_agent_scene()])
    monkeypatch.setattr(pipeline, "make_scene_factory", lambda scenes, selection, fixed_scene_index, seed: lambda: scenes[0])
    monkeypatch.setattr(pipeline, "TorchORCAEnv", lambda scene_factory, config: object())

    cfg = SkrlEnvBuildConfig(map_extractor_type="vae_tap", vae_checkpoint=None)
    with pytest.raises(ValueError, match="vae_checkpoint is required"):
        pipeline._make_single_env(cfg, seed=0, device=torch.device("cpu"))


def test_run_skrl_ppo_training_happy_path(monkeypatch, tmp_path: Path) -> None:
    events: dict[str, object] = {}

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.closed = False

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}

        def close(self):
            self.closed = True

    class _DummyWrappedEnv(_DummyEnv):
        num_envs = 1

    class _DummyAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events["agent_cfg"] = kwargs.get("cfg")

        def save(self, path: str) -> None:
            Path(path).write_text("ok", encoding="utf-8")

    class _DummyTrainer:
        def __init__(self, env, agents, cfg):
            self.env = env
            self.agents = agents
            self.cfg = cfg

        def train(self) -> None:
            events["trained"] = True

    dummy_wrapped = _DummyWrappedEnv()

    monkeypatch.setattr(pipeline, "seed_everything", lambda seed: events.setdefault("seed", seed))
    monkeypatch.setattr(pipeline, "_make_single_env", lambda env_config, seed, device: _DummyEnv())
    monkeypatch.setattr(pipeline, "PeriodicEpisodeSummaryWrapper", lambda env, interval_episodes, prefix: env)
    monkeypatch.setattr(pipeline, "wrap_env", lambda env, wrapper, verbose=False: dummy_wrapped)
    def _fake_policy_model(*args, **kwargs):
        del args
        events["policy_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(pipeline, "OccupancyPolicyModel", _fake_policy_model)
    monkeypatch.setattr(pipeline, "OccupancyValueModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "RandomMemory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "PPO", _DummyAgent)
    monkeypatch.setattr(pipeline, "install_agent_tracking_summary", lambda agent, prefix: events.setdefault("summary_hook", True))
    monkeypatch.setattr(pipeline, "SequentialTrainer", _DummyTrainer)

    env_cfg = SkrlEnvBuildConfig()
    out = tmp_path / "model.pt"
    train_cfg = SkrlPPOTrainConfig(
        total_timesteps=8,
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        num_envs=1,
        device="cpu",
        initial_policy_std=0.7,
        max_policy_std=0.9,
        entropy_loss_scale=0.02,
        output=out,
    )

    returned = pipeline.run_skrl_ppo_training(env_cfg, train_cfg)

    assert returned == out
    assert out.exists()
    assert events.get("trained") is True
    assert dummy_wrapped.closed is True
    assert events["policy_kwargs"]["initial_std"] == pytest.approx(0.7)
    assert events["policy_kwargs"]["max_std"] == pytest.approx(0.9)
    assert events["agent_cfg"]["entropy_loss_scale"] == pytest.approx(0.02)


def test_run_skrl_ppo_training_validates_inputs() -> None:
    env_cfg = SkrlEnvBuildConfig()

    with pytest.raises(NotImplementedError, match="supports only num_envs=1"):
        pipeline.run_skrl_ppo_training(
            env_cfg,
            SkrlPPOTrainConfig(num_envs=2, device="cpu"),
        )

    with pytest.raises(ValueError, match="summary_interval_episodes must be > 0"):
        pipeline.run_skrl_ppo_training(
            env_cfg,
            SkrlPPOTrainConfig(num_envs=1, device="cpu", summary_interval_episodes=0),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
