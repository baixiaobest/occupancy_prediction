from __future__ import annotations

import copy
from typing import Mapping, Optional, Tuple, Union

import gymnasium
import torch

from skrl.agents.torch import Agent
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class CustomSAC(SAC):
    """SAC variant that updates every ``train_freq`` interactions after ``learning_starts``."""

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        _cfg.setdefault("train_freq", 1)
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self._train_freq = int(self.cfg.get("train_freq", 1))
        if self._train_freq <= 0:
            raise ValueError("train_freq must be > 0")

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Update every ``train_freq`` steps once ``learning_starts`` is reached."""
        should_update = (
            int(timestep) >= int(self._learning_starts)
            and (int(timestep) - int(self._learning_starts) + 1) % int(self._train_freq) == 0
        )
        if should_update:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        Agent.post_interaction(self, timestep, timesteps)
