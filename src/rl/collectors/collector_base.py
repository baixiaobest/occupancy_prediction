from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.scene import Scene

from ..managers.observation_manager import ObservationBatchContext, ObservationManager, build_observation_manager
from ..replay_buffer import ReplayBuffer, TensorDict


class BaseActionCollector(ABC):
    def __init__(
        self,
        *,
        env: object,
        replay_buffer: ReplayBuffer,
        observation_manager: ObservationManager | None,
        seed: int,
        selection_seed: int | None = None,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.observation_manager = self._resolve_observation_manager(observation_manager)
        self._prepared_obs: TensorDict | None = None
        self._seed = int(seed)
        self._selection_seed = None if selection_seed is None else int(selection_seed)
        self._rng: torch.Generator | None = None
        self._rng_device: torch.device | None = None
        self._selection_rng: torch.Generator | None = None
        self._selection_rng_device: torch.device | None = None

    def reset_episode(self, seed: int | None = None) -> TensorDict:
        self.observation_manager.reset()
        raw_obs = self.env.reset(seed=seed)
        scene = self._get_scene()
        self._prepared_obs = self.observation_manager.compute(
            ObservationBatchContext(raw_obs=raw_obs, scene=scene)
        )
        return self._prepared_obs

    def collect_steps(self, num_steps: int, *, reset_seed: int | None = None):
        steps = int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self._prepared_obs is None:
            self.reset_episode(seed=reset_seed)

        transitions_added = 0
        episodes_completed = 0
        total_reward = 0.0
        for _ in range(steps):
            if self._prepared_obs is None:
                raise RuntimeError("Collector observation state is not initialized")

            obs = self._prepared_obs
            action_selection = self.select_action(obs)
            next_raw_obs, rewards, dones, _infos = self._step_env(action_selection)
            next_obs = self.prepare_observation(next_raw_obs)

            self.replay_buffer.add_batch(
                obs=obs,
                actions=self._actions_for_replay(action_selection),
                rewards=torch.as_tensor(rewards, dtype=torch.float32),
                next_obs=next_obs,
                dones=torch.as_tensor(dones),
                candidate_actions=self._candidate_actions_for_replay(action_selection),
                candidate_log_probs=getattr(action_selection, "candidate_log_probs", None),
            )

            transitions_added += 1
            total_reward += float(torch.as_tensor(rewards, dtype=torch.float32).sum().item())
            if bool(torch.as_tensor(dones).reshape(-1)[0].item()):
                episodes_completed += 1
                self.reset_episode()
            else:
                self._prepared_obs = next_obs

        return self._build_collect_summary(
            transitions_added=transitions_added,
            episodes_completed=episodes_completed,
            total_reward=total_reward,
        )

    def prepare_observation(self, raw_obs: TensorDict) -> TensorDict:
        self._prepared_obs = self.observation_manager.compute(
            ObservationBatchContext(raw_obs=raw_obs, scene=self._get_scene())
        )
        return self._prepared_obs

    def _get_rng(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self._seed)
            self._rng_device = device
        return self._rng

    def _get_selection_rng(self, device: torch.device) -> torch.Generator:
        if self._selection_seed is None:
            raise RuntimeError("selection RNG requested without selection seed")
        if self._selection_rng is None or self._selection_rng_device != device:
            self._selection_rng = torch.Generator(device=device)
            self._selection_rng.manual_seed(self._selection_seed)
            self._selection_rng_device = device
        return self._selection_rng

    def _get_scene(self) -> Scene:
        sim = getattr(self.env, "sim", None)
        scene = getattr(sim, "scene", None)
        if not isinstance(scene, Scene):
            raise RuntimeError("collector env must expose env.sim.scene as a Scene")
        return scene

    def _resolve_observation_manager(
        self,
        observation_manager: ObservationManager | None,
    ) -> ObservationManager:
        if observation_manager is not None:
            return observation_manager

        env_config = getattr(self.env, "env_config", None)
        observation_config = getattr(env_config, "observation", None)
        if observation_config is None:
            raise ValueError(
                "observation_manager must be provided when env.env_config.observation is not set"
            )
        return build_observation_manager(observation_config)

    @abstractmethod
    def select_action(self, obs: TensorDict):
        raise NotImplementedError

    @abstractmethod
    def _step_env(self, action_selection):
        raise NotImplementedError

    @abstractmethod
    def _actions_for_replay(self, action_selection) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _candidate_actions_for_replay(self, action_selection) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _build_collect_summary(
        self,
        *,
        transitions_added: int,
        episodes_completed: int,
        total_reward: float,
    ):
        raise NotImplementedError


# Backward-compatible alias while migrating call sites.
BaseRandomActionCollector = BaseActionCollector