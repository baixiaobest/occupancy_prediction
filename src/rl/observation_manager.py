from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.occupancy2d import Occupancy2d
from src.scene import Scene

from .replay_buffer import TensorDict


@dataclass
class ObservationBatchContext:
    """Batched observation computation context.

    Shapes:
    - raw_obs["positions"]: (N_env, N_agents, 2)
    - raw_obs["velocities"]: (N_env, N_agents, 2)
    - raw_obs["goals"]: (N_env, N_agents, 2)
    - raw_obs["controlled_agent_index"]: (N_env,)
    """

    raw_obs: TensorDict
    scene: Scene | list[Scene]
    extras: dict[str, Any] = field(default_factory=dict)
    positions: torch.Tensor = field(init=False)
    velocities: torch.Tensor = field(init=False)
    goals: torch.Tensor = field(init=False)
    controlled_agent_indices: torch.Tensor = field(init=False)
    scenes: list[Scene] = field(init=False)

    def __post_init__(self) -> None:
        required_keys = {"positions", "velocities", "goals", "controlled_agent_index"}
        missing = required_keys.difference(self.raw_obs.keys())
        if missing:
            raise ValueError(f"raw_obs is missing required keys: {sorted(missing)}")

        self.positions = torch.as_tensor(self.raw_obs["positions"], dtype=torch.float32)
        device = self.positions.device
        self.velocities = torch.as_tensor(self.raw_obs["velocities"], dtype=torch.float32, device=device)
        self.goals = torch.as_tensor(self.raw_obs["goals"], dtype=torch.float32, device=device)
        self.controlled_agent_indices = torch.as_tensor(
            self.raw_obs["controlled_agent_index"],
            dtype=torch.int64,
            device=device,
        ).reshape(-1)

        if self.positions.ndim != 3 or self.positions.shape[-1] != 2:
            raise ValueError("raw_obs['positions'] must have shape (N_env, N_agents, 2)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("raw_obs['velocities'] must match raw_obs['positions'] shape")
        if self.goals.shape != self.positions.shape:
            raise ValueError("raw_obs['goals'] must match raw_obs['positions'] shape")

        n_env, n_agents, _ = self.positions.shape
        if self.controlled_agent_indices.shape != (n_env,):
            raise ValueError("raw_obs['controlled_agent_index'] must have shape (N_env,)")
        if bool(torch.any(self.controlled_agent_indices < 0)) or bool(
            torch.any(self.controlled_agent_indices >= n_agents)
        ):
            raise ValueError("controlled_agent_index contains out-of-range index")

        if isinstance(self.scene, Scene):
            self.scenes = [self.scene]
        else:
            self.scenes = list(self.scene)
        if len(self.scenes) != n_env:
            raise ValueError(f"Expected {n_env} scenes, got {len(self.scenes)}")

    @property
    def num_envs(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.positions.shape[1])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def controlled_positions(self) -> torch.Tensor:
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self.positions[env_idx, self.controlled_agent_indices]

    def controlled_velocities(self) -> torch.Tensor:
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self.velocities[env_idx, self.controlled_agent_indices]

    def controlled_goals(self) -> torch.Tensor:
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self.goals[env_idx, self.controlled_agent_indices]

    def controlled_goal_offsets(self) -> torch.Tensor:
        return self.controlled_goals() - self.controlled_positions()


ObservationTermFn = Callable[[ObservationBatchContext, dict[str, Any]], torch.Tensor]


@dataclass
class ObservationTermCfg:
    """Configuration for one observation term."""

    name: str
    fn: ObservationTermFn
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ObservationConfig:
    """Configuration wrapper for observation-manager terms."""

    terms: list[ObservationTermCfg] = field(default_factory=list)


class ObservationManager:
    """Aggregates observation terms over a batch of environments."""

    def __init__(self, terms: list[ObservationTermCfg] | None = None) -> None:
        self.terms: list[ObservationTermCfg] = []
        self._state: dict[str, Any] = {}
        if terms is not None:
            for term in terms:
                self.add_term(term)

    def add_term(self, term: ObservationTermCfg) -> None:
        if not term.name:
            raise ValueError("Observation term name must not be empty")
        if not callable(term.fn):
            raise ValueError(f"Observation term {term.name} fn must be callable")
        self.terms.append(term)

    def reset(self) -> None:
        self._state.clear()

    def compute(self, context: ObservationBatchContext) -> TensorDict:
        context.extras["manager_state"] = self._state
        observations: TensorDict = {}

        for term in self.terms:
            if not term.enabled:
                continue

            value = torch.as_tensor(term.fn(context, term.params), dtype=torch.float32)
            if value.ndim < 1 or int(value.shape[0]) != context.num_envs:
                raise ValueError(
                    f"Observation term {term.name} must return leading shape (N_env, ...), got {tuple(value.shape)}"
                )
            observations[term.name] = value

        return observations


@dataclass
class OnlineOccupancyObservationConfig:
    decoder_context_len: int
    local_map_shape: tuple[int, int]
    occupancy_resolution: tuple[float, float]
    agent_radius: float = 0.3
    device: str = "cpu"


def _build_map_size_xy(
    local_map_shape: tuple[int, int],
    occupancy_resolution: tuple[float, float],
) -> tuple[float, float]:
    patch_h = int(local_map_shape[0])
    patch_w = int(local_map_shape[1])
    res_x = float(occupancy_resolution[0])
    res_y = float(occupancy_resolution[1])
    return patch_w * res_x, patch_h * res_y


def _get_manager_state(context: ObservationBatchContext) -> dict[str, Any]:
    state = context.extras.get("manager_state")
    if not isinstance(state, dict):
        raise RuntimeError("Observation manager state is not initialized")
    return state


def _get_or_create_state_list(
    state: dict[str, Any],
    *,
    key: str,
    length: int,
    factory: Callable[[], Any],
) -> list[Any]:
    values = state.get(key)
    if not isinstance(values, list) or len(values) != length:
        values = [factory() for _ in range(length)]
        state[key] = values
    return values


def term_controlled_current_velocity(
    context: ObservationBatchContext,
    params: dict[str, Any],
) -> torch.Tensor:
    del params
    return context.controlled_velocities()


def term_controlled_goal_offset(
    context: ObservationBatchContext,
    params: dict[str, Any],
) -> torch.Tensor:
    del params
    return context.controlled_goal_offsets()


def term_static_local_occupancy(
    context: ObservationBatchContext,
    params: dict[str, Any],
) -> torch.Tensor:
    device = torch.device(params.get("device", context.device))
    resolution = tuple(float(v) for v in params["occupancy_resolution"])
    local_map_shape = tuple(int(v) for v in params["local_map_shape"])
    agent_radius = float(params.get("agent_radius", 0.3))
    state_key = str(params.get("state_key", "static_occupancy_renderers"))

    state = _get_manager_state(context)
    renderers = _get_or_create_state_list(
        state,
        key=state_key,
        length=context.num_envs,
        factory=lambda: Occupancy2d(
            resolution=resolution,
            size=_build_map_size_xy(local_map_shape, resolution),
            agent_radius=agent_radius,
        ),
    )

    outputs: list[torch.Tensor] = []
    controlled_positions = context.controlled_positions()
    for env_idx, renderer in enumerate(renderers):
        renderer.update_inputs(trajectory=None, static_obstacles=context.scenes[env_idx].obstacles)
        grid = renderer.generate(center_offset=controlled_positions[env_idx])[0]
        outputs.append(grid.to(device=device, dtype=torch.float32).unsqueeze(0))

    return torch.stack(outputs, dim=0)


def term_dynamic_local_occupancy_context(
    context: ObservationBatchContext,
    params: dict[str, Any],
) -> torch.Tensor:
    device = torch.device(params.get("device", context.device))
    resolution = tuple(float(v) for v in params["occupancy_resolution"])
    local_map_shape = tuple(int(v) for v in params["local_map_shape"])
    agent_radius = float(params.get("agent_radius", 0.3))
    context_len = int(params["decoder_context_len"])
    renderer_key = str(params.get("renderer_key", "dynamic_occupancy_renderers"))
    history_key = str(params.get("history_key", "dynamic_occupancy_history"))

    state = _get_manager_state(context)
    renderers = _get_or_create_state_list(
        state,
        key=renderer_key,
        length=context.num_envs,
        factory=lambda: Occupancy2d(
            resolution=resolution,
            size=_build_map_size_xy(local_map_shape, resolution),
            agent_radius=agent_radius,
        ),
    )
    histories = _get_or_create_state_list(
        state,
        key=history_key,
        length=context.num_envs,
        factory=lambda: deque(maxlen=context_len + 1),
    )

    outputs: list[torch.Tensor] = []
    controlled_positions = context.controlled_positions()
    for env_idx, renderer in enumerate(renderers):
        renderer.update_inputs(trajectory=context.positions[env_idx].unsqueeze(0), static_obstacles=[])
        frame = renderer.generate(center_offset=controlled_positions[env_idx])[0].to(device=device, dtype=torch.float32)
        histories[env_idx].append(frame)

        context_frames = list(histories[env_idx])[:-1]
        if len(context_frames) == 0:
            context_frames = [frame]
        context_frames = context_frames[-context_len:]
        while len(context_frames) < context_len:
            context_frames.insert(0, context_frames[0])

        outputs.append(torch.stack(context_frames, dim=0).unsqueeze(0))

    return torch.stack(outputs, dim=0)


def build_observation_manager(config: ObservationConfig) -> ObservationManager:
    return ObservationManager(terms=config.terms)


def build_online_occupancy_observation_config(
    config: OnlineOccupancyObservationConfig,
) -> ObservationConfig:
    if int(config.decoder_context_len) <= 0:
        raise ValueError("decoder_context_len must be > 0")
    if len(config.local_map_shape) != 2 or any(int(v) <= 0 for v in config.local_map_shape):
        raise ValueError("local_map_shape must contain two positive ints")
    if len(config.occupancy_resolution) != 2 or any(float(v) <= 0.0 for v in config.occupancy_resolution):
        raise ValueError("occupancy_resolution must contain two positive floats")
    if float(config.agent_radius) < 0.0:
        raise ValueError("agent_radius must be >= 0")

    shared_params = {
        "decoder_context_len": int(config.decoder_context_len),
        "local_map_shape": tuple(int(v) for v in config.local_map_shape),
        "occupancy_resolution": tuple(float(v) for v in config.occupancy_resolution),
        "agent_radius": float(config.agent_radius),
        "device": str(config.device),
    }
    return ObservationConfig(
        terms=[
            ObservationTermCfg(
                name="dynamic_context",
                fn=term_dynamic_local_occupancy_context,
                params={
                    **shared_params,
                    "renderer_key": "dynamic_context_renderer",
                    "history_key": "dynamic_context_history",
                },
            ),
            ObservationTermCfg(
                name="static_map",
                fn=term_static_local_occupancy,
                params={
                    **shared_params,
                    "state_key": "static_map_renderer",
                },
            ),
            ObservationTermCfg(name="current_velocity", fn=term_controlled_current_velocity),
            ObservationTermCfg(name="goal_position", fn=term_controlled_goal_offset),
        ]
    )


def build_online_occupancy_observation_manager(
    config: OnlineOccupancyObservationConfig,
) -> ObservationManager:
    return build_observation_manager(build_online_occupancy_observation_config(config))


__all__ = [
    "ObservationBatchContext",
    "ObservationConfig",
    "ObservationManager",
    "ObservationTermCfg",
    "ObservationTermFn",
    "OnlineOccupancyObservationConfig",
    "build_observation_manager",
    "build_online_occupancy_observation_config",
    "build_online_occupancy_observation_manager",
    "term_controlled_current_velocity",
    "term_controlled_goal_offset",
    "term_dynamic_local_occupancy_context",
    "term_static_local_occupancy",
]