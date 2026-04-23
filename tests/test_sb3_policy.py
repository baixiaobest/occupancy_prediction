from __future__ import annotations

import numpy as np

import importlib

pytest = importlib.import_module("pytest")

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import gymnasium as gym

from sb3.policy import OccupancyActorCriticPolicy


class _DummyOccupancyDictEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "dynamic_context": gym.spaces.Box(low=0.0, high=1.0, shape=(1, 4, 16, 16), dtype=np.float32),
                "static_map": gym.spaces.Box(low=0.0, high=1.0, shape=(1, 16, 16), dtype=np.float32),
                "current_velocity": gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
                "goal_position": gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )
        self._rng = np.random.default_rng(0)
        self._steps = 0

    def _sample_obs(self) -> dict[str, np.ndarray]:
        return {
            "dynamic_context": self._rng.random((1, 4, 16, 16), dtype=np.float32),
            "static_map": self._rng.random((1, 16, 16), dtype=np.float32),
            "current_velocity": self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32),
            "goal_position": self._rng.uniform(-3.0, 3.0, size=(2,)).astype(np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._steps = 0
        return self._sample_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        del action
        self._steps += 1
        terminated = False
        truncated = self._steps >= 8
        reward = 0.0
        return self._sample_obs(), float(reward), bool(terminated), bool(truncated), {}


def test_occupancy_actor_critic_policy_short_rollout() -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    ppo_cls = sb3.PPO

    env = _DummyOccupancyDictEnv()
    model = ppo_cls(
        policy=OccupancyActorCriticPolicy,
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
