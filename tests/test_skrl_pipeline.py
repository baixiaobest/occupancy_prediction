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
from src.skrl.config import SkrlEnvBuildConfig, SkrlPPOTrainConfig, SkrlSACTrainConfig
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
    monkeypatch.setattr(
        pipeline,
        "PeriodicEpisodeSummaryWrapper",
        lambda env, interval_episodes, prefix, **kwargs: env,
    )
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
    assert events["agent_cfg"]["experiment"]["wandb"] is False


def test_run_skrl_ppo_training_forwards_wandb_settings(monkeypatch, tmp_path: Path) -> None:
    events: dict[str, object] = {}

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}

        def close(self):
            return None

    class _DummyWrappedEnv(_DummyEnv):
        num_envs = 1

    class _DummyAgent:
        def __init__(self, **kwargs):
            events["agent_cfg"] = kwargs.get("cfg")

        def save(self, path: str) -> None:
            Path(path).write_text("ok", encoding="utf-8")

    class _DummyTrainer:
        def __init__(self, env, agents, cfg):
            del env, agents, cfg

        def train(self) -> None:
            events["trained"] = True

    dummy_wrapped = _DummyWrappedEnv()

    monkeypatch.setattr(pipeline, "seed_everything", lambda seed: events.setdefault("seed", seed))
    monkeypatch.setattr(pipeline, "_make_single_env", lambda env_config, seed, device: _DummyEnv())
    monkeypatch.setattr(
        pipeline,
        "PeriodicEpisodeSummaryWrapper",
        lambda env, interval_episodes, prefix, **kwargs: env,
    )
    monkeypatch.setattr(pipeline, "wrap_env", lambda env, wrapper, verbose=False: dummy_wrapped)
    monkeypatch.setattr(pipeline, "OccupancyPolicyModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "OccupancyValueModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "RandomMemory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "PPO", _DummyAgent)
    monkeypatch.setattr(pipeline, "install_agent_tracking_summary", lambda agent, prefix: None)
    monkeypatch.setattr(pipeline, "SequentialTrainer", _DummyTrainer)
    monkeypatch.setattr(
        pipeline,
        "_save_checkpoint_to_wandb_if_enabled",
        lambda *, train_config, checkpoint_path: events.setdefault("wandb_checkpoint", str(checkpoint_path)),
    )

    env_cfg = SkrlEnvBuildConfig()
    out = tmp_path / "wandb_model.pt"
    train_cfg = SkrlPPOTrainConfig(
        total_timesteps=8,
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        num_envs=1,
        device="cpu",
        wandb=True,
        wandb_project="unit-test-project",
        wandb_run_name=None,
        output=out,
    )

    pipeline.run_skrl_ppo_training(env_cfg, train_cfg)

    assert events.get("trained") is True
    experiment_cfg = events["agent_cfg"]["experiment"]
    assert experiment_cfg["wandb"] is True
    assert experiment_cfg["wandb_kwargs"] == {
        "project": "unit-test-project",
        "name": "wandb_model",
    }
    assert events["wandb_checkpoint"] == str(out)


def test_run_skrl_ppo_training_shares_tap_projector(monkeypatch, tmp_path: Path) -> None:
    events: dict[str, object] = {}

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict(
                {
                    "decoder_tap": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
                    "goal_position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "current_velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "last_commanded_velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        def reset(self, *, seed=None, options=None):
            del seed, options
            obs = {
                "decoder_tap": np.zeros((10,), dtype=np.float32),
                "goal_position": np.zeros((2,), dtype=np.float32),
                "current_velocity": np.zeros((2,), dtype=np.float32),
                "last_commanded_velocity": np.zeros((2,), dtype=np.float32),
            }
            return obs, {}

        def step(self, action):
            del action
            obs = {
                "decoder_tap": np.zeros((10,), dtype=np.float32),
                "goal_position": np.zeros((2,), dtype=np.float32),
                "current_velocity": np.zeros((2,), dtype=np.float32),
                "last_commanded_velocity": np.zeros((2,), dtype=np.float32),
            }
            return obs, 0.0, False, False, {}

        def close(self):
            return None

    class _DummyWrappedEnv(_DummyEnv):
        num_envs = 1

    class _DummyAgent:
        def __init__(self, **kwargs):
            events["agent_cfg"] = kwargs.get("cfg")

        def save(self, path: str) -> None:
            Path(path).write_text("ok", encoding="utf-8")

    class _DummyTrainer:
        def __init__(self, env, agents, cfg):
            del env, agents, cfg

        def train(self) -> None:
            events["trained"] = True

    dummy_wrapped = _DummyWrappedEnv()

    monkeypatch.setattr(pipeline, "seed_everything", lambda seed: events.setdefault("seed", seed))
    monkeypatch.setattr(pipeline, "_make_single_env", lambda env_config, seed, device: _DummyEnv())
    monkeypatch.setattr(
        pipeline,
        "PeriodicEpisodeSummaryWrapper",
        lambda env, interval_episodes, prefix, **kwargs: env,
    )
    monkeypatch.setattr(pipeline, "wrap_env", lambda env, wrapper, verbose=False: dummy_wrapped)

    shared_projector = object()

    def _fake_build_tap_projector(observation_space, **kwargs):
        del observation_space
        events["tap_builder_hidden"] = kwargs.get("tap_bottleneck_hidden_dims")
        events["tap_builder_output"] = kwargs.get("tap_bottleneck_output_dim")
        return shared_projector

    def _fake_policy_model(*args, **kwargs):
        del args
        events["policy_tap_projector"] = kwargs.get("tap_projector")
        return object()

    def _fake_value_model(*args, **kwargs):
        del args
        events["value_tap_projector"] = kwargs.get("tap_projector")
        return object()

    monkeypatch.setattr(pipeline, "build_tap_bottleneck_feature_projector", _fake_build_tap_projector)
    monkeypatch.setattr(pipeline, "OccupancyPolicyModel", _fake_policy_model)
    monkeypatch.setattr(pipeline, "OccupancyValueModel", _fake_value_model)
    monkeypatch.setattr(pipeline, "RandomMemory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "PPO", _DummyAgent)
    monkeypatch.setattr(pipeline, "install_agent_tracking_summary", lambda agent, prefix: None)
    monkeypatch.setattr(pipeline, "SequentialTrainer", _DummyTrainer)

    env_cfg = SkrlEnvBuildConfig(
        observation_mode="occupancy",
        map_extractor_type="vae_tap",
        vae_checkpoint=tmp_path / "vae.pt",
        vae_tap_layer=3,
    )
    out = tmp_path / "model.pt"
    train_cfg = SkrlPPOTrainConfig(
        total_timesteps=8,
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        num_envs=1,
        device="cpu",
        tap_bottleneck_hidden_dims=(64, 32),
        tap_bottleneck_output_dim=16,
        output=out,
    )

    pipeline.run_skrl_ppo_training(env_cfg, train_cfg)

    assert events.get("trained") is True
    assert events["tap_builder_hidden"] == (64, 32)
    assert events["tap_builder_output"] == 16
    assert events["policy_tap_projector"] is shared_projector
    assert events["value_tap_projector"] is shared_projector


def test_run_skrl_ppo_training_uses_vector_builder_for_multi_env(monkeypatch, tmp_path: Path) -> None:
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
        num_envs = 2

    class _DummyAgent:
        def __init__(self, **kwargs):
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
    monkeypatch.setattr(pipeline, "_build_scene_pool", lambda config, seed: [_single_agent_scene()])

    def _fake_build_vec_env(**kwargs):
        events["vec_builder_kwargs"] = kwargs
        return _DummyEnv()

    monkeypatch.setattr(pipeline, "build_torch_orca_vec_env", _fake_build_vec_env)
    monkeypatch.setattr(pipeline, "wrap_env", lambda env, wrapper, verbose=False: dummy_wrapped)
    monkeypatch.setattr(pipeline, "OccupancyPolicyModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "OccupancyValueModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "RandomMemory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "PPO", _DummyAgent)
    monkeypatch.setattr(pipeline, "install_agent_tracking_summary", lambda agent, prefix: None)
    monkeypatch.setattr(pipeline, "SequentialTrainer", _DummyTrainer)

    env_cfg = SkrlEnvBuildConfig(observation_mode="occupancy")
    out = tmp_path / "model.pt"
    train_cfg = SkrlPPOTrainConfig(
        total_timesteps=8,
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        num_envs=2,
        vec_env_backend="torch_dummy",
        device="cpu",
        output=out,
    )

    returned = pipeline.run_skrl_ppo_training(env_cfg, train_cfg)

    assert returned == out
    assert out.exists()
    assert events.get("trained") is True
    assert dummy_wrapped.closed is True
    assert int(events["vec_builder_kwargs"]["num_envs"]) == 2
    assert events["vec_builder_kwargs"]["backend"] == "torch_dummy"
    assert events["vec_builder_kwargs"]["observation_mode"] == "occupancy"


def test_run_skrl_ppo_training_validates_inputs() -> None:
    env_cfg = SkrlEnvBuildConfig()

    with pytest.raises(ValueError, match="num_envs must be > 0"):
        pipeline.run_skrl_ppo_training(
            env_cfg,
            SkrlPPOTrainConfig(num_envs=0, device="cpu"),
        )

    with pytest.raises(ValueError, match="summary_interval_episodes must be > 0"):
        pipeline.run_skrl_ppo_training(
            env_cfg,
            SkrlPPOTrainConfig(num_envs=1, device="cpu", summary_interval_episodes=0),
        )


def test_run_skrl_sac_training_forwards_train_freq(monkeypatch, tmp_path: Path) -> None:
    events: dict[str, object] = {}

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            del action
            return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}

        def close(self):
            return None

    class _DummyWrappedEnv(_DummyEnv):
        num_envs = 1

    class _DummyAgent:
        def __init__(self, **kwargs):
            events["agent_cfg"] = kwargs.get("cfg")

        def save(self, path: str) -> None:
            Path(path).write_text("ok", encoding="utf-8")

    class _DummyTrainer:
        def __init__(self, env, agents, cfg):
            del env, agents, cfg

        def train(self) -> None:
            events["trained"] = True

    dummy_wrapped = _DummyWrappedEnv()

    monkeypatch.setattr(pipeline, "seed_everything", lambda seed: events.setdefault("seed", seed))
    monkeypatch.setattr(pipeline, "_make_single_env", lambda env_config, seed, device: _DummyEnv())
    monkeypatch.setattr(
        pipeline,
        "PeriodicEpisodeSummaryWrapper",
        lambda env, interval_episodes, prefix, **kwargs: env,
    )
    monkeypatch.setattr(pipeline, "wrap_env", lambda env, wrapper, verbose=False: dummy_wrapped)
    monkeypatch.setattr(pipeline, "OccupancyPolicyModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "OccupancyQValueModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "RandomMemory", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "SACWithTrainFreq", _DummyAgent)
    monkeypatch.setattr(pipeline, "SequentialTrainer", _DummyTrainer)

    env_cfg = SkrlEnvBuildConfig()
    out = tmp_path / "sac_model.pt"
    train_cfg = SkrlSACTrainConfig(
        total_timesteps=8,
        num_envs=1,
        device="cpu",
        train_freq=7,
        output=out,
    )

    returned = pipeline.run_skrl_sac_training(env_cfg, train_cfg)

    assert returned == out
    assert out.exists()
    assert events.get("trained") is True
    assert events["agent_cfg"]["train_freq"] == 7


def test_run_skrl_sac_training_validates_train_freq() -> None:
    env_cfg = SkrlEnvBuildConfig()

    with pytest.raises(ValueError, match="train_freq must be > 0"):
        pipeline.run_skrl_sac_training(
            env_cfg,
            SkrlSACTrainConfig(train_freq=0, device="cpu"),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
