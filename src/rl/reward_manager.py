from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class RewardBatchContext:
    """Batched reward computation context.

    Shapes:
    - prev_positions: (N_env, N_agents, 2)
    - new_positions: (N_env, N_agents, 2)
    - goals: (N_env, N_agents, 2)
    - controlled_agent_indices: (N_env,)
    - goal_tolerances: (N_env,)
    """

    prev_positions: torch.Tensor
    new_positions: torch.Tensor
    goals: torch.Tensor
    controlled_agent_indices: torch.Tensor
    goal_tolerances: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prev_positions = torch.as_tensor(self.prev_positions, dtype=torch.float32)
        device = self.prev_positions.device
        self.new_positions = torch.as_tensor(self.new_positions, dtype=torch.float32, device=device)
        self.goals = torch.as_tensor(self.goals, dtype=torch.float32, device=device)
        self.controlled_agent_indices = torch.as_tensor(
            self.controlled_agent_indices,
            dtype=torch.int64,
            device=device,
        )
        self.goal_tolerances = torch.as_tensor(
            self.goal_tolerances,
            dtype=torch.float32,
            device=device,
        )

        if self.prev_positions.ndim != 3 or self.prev_positions.shape[-1] != 2:
            raise ValueError("prev_positions must have shape (N_env, N_agents, 2)")
        if self.new_positions.shape != self.prev_positions.shape:
            raise ValueError("new_positions must match prev_positions shape")
        if self.goals.shape != self.prev_positions.shape:
            raise ValueError("goals must match prev_positions shape")

        n_env, n_agents, _ = self.prev_positions.shape
        if self.controlled_agent_indices.shape != (n_env,):
            raise ValueError("controlled_agent_indices must have shape (N_env,)")
        if self.goal_tolerances.shape != (n_env,):
            raise ValueError("goal_tolerances must have shape (N_env,)")

        if bool(torch.any(self.controlled_agent_indices < 0)) or bool(
            torch.any(self.controlled_agent_indices >= n_agents)
        ):
            raise ValueError("controlled_agent_indices contains out-of-range index")

    @property
    def num_envs(self) -> int:
        return int(self.prev_positions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.prev_positions.shape[1])

    def controlled_prev_positions(self) -> torch.Tensor:
        idx = torch.arange(self.num_envs, device=self.prev_positions.device)
        return self.prev_positions[idx, self.controlled_agent_indices]

    def controlled_new_positions(self) -> torch.Tensor:
        idx = torch.arange(self.num_envs, device=self.new_positions.device)
        return self.new_positions[idx, self.controlled_agent_indices]

    def controlled_goals(self) -> torch.Tensor:
        idx = torch.arange(self.num_envs, device=self.goals.device)
        return self.goals[idx, self.controlled_agent_indices]

    def controlled_prev_goal_distance(self) -> torch.Tensor:
        diff = self.controlled_goals() - self.controlled_prev_positions()
        return torch.linalg.vector_norm(diff, dim=1)

    def controlled_new_goal_distance(self) -> torch.Tensor:
        diff = self.controlled_goals() - self.controlled_new_positions()
        return torch.linalg.vector_norm(diff, dim=1)


RewardTermFn = Callable[[RewardBatchContext, dict[str, Any]], torch.Tensor]


@dataclass
class RewardTermCfg:
    """Configuration for one reward term.

    The final contribution is: weight * fn(context, params).
    """

    name: str
    fn: RewardTermFn
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


def term_progress_to_goal(context: RewardBatchContext, params: dict[str, Any]) -> torch.Tensor:
    """Positive when controlled agent gets closer to its goal."""
    del params
    return context.controlled_prev_goal_distance() - context.controlled_new_goal_distance()


def term_collision_any(context: RewardBatchContext, params: dict[str, Any]) -> torch.Tensor:
    """Binary collision indicator for the controlled agent against any other agent."""
    collision_distance = float(params.get("collision_distance", 0.4))

    controlled_pos = context.controlled_new_positions()
    diffs = context.new_positions - controlled_pos[:, None, :]
    dists = torch.linalg.vector_norm(diffs, dim=2)

    self_mask = torch.zeros(
        (context.num_envs, context.num_agents),
        dtype=torch.bool,
        device=context.new_positions.device,
    )
    self_mask[
        torch.arange(context.num_envs, device=context.new_positions.device),
        context.controlled_agent_indices,
    ] = True

    collision = torch.any((dists < collision_distance) & (~self_mask), dim=1)
    return collision.to(dtype=torch.float32)


def term_success(context: RewardBatchContext, params: dict[str, Any]) -> torch.Tensor:
    """Binary success indicator: controlled agent reached goal tolerance."""
    del params
    goal_dist = context.controlled_new_goal_distance()
    return (goal_dist <= context.goal_tolerances).to(dtype=torch.float32)


def term_constant(context: RewardBatchContext, params: dict[str, Any]) -> torch.Tensor:
    """Constant scalar term per environment (useful for step penalty)."""
    value = float(params.get("value", 1.0))
    return torch.full(
        (context.num_envs,),
        fill_value=value,
        dtype=torch.float32,
        device=context.prev_positions.device,
    )


def _default_reward_terms() -> list[RewardTermCfg]:
    return [
        RewardTermCfg(
            name="progress",
            fn=term_progress_to_goal,
            weight=1.0,
        ),
        RewardTermCfg(
            name="step_penalty",
            fn=term_constant,
            weight=0.0,
            params={"value": 1.0},
        ),
        RewardTermCfg(
            name="collision",
            fn=term_collision_any,
            weight=-1.0,
            params={"collision_distance": 0.4},
        ),
        RewardTermCfg(
            name="success",
            fn=term_success,
            weight=5.0,
        ),
    ]


@dataclass
class RewardConfig:
    """Configuration wrapper for reward-manager terms."""

    terms: list[RewardTermCfg] = field(default_factory=_default_reward_terms)


class RewardManager:
    """Aggregates weighted reward terms over a batch of environments."""

    def __init__(self, terms: list[RewardTermCfg] | None = None) -> None:
        self.terms: list[RewardTermCfg] = []
        if terms is not None:
            for term in terms:
                self.add_term(term)

    def add_term(self, term: RewardTermCfg) -> None:
        if not term.name:
            raise ValueError("Reward term name must not be empty")
        if not callable(term.fn):
            raise ValueError(f"Reward term {term.name} fn must be callable")
        self.terms.append(term)

    def compute(
        self,
        context: RewardBatchContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute total reward and per-term breakdown.

        Returns:
            total_reward: (N_env,)
            weighted_terms: dict[name -> (N_env,)]
            raw_terms: dict[name -> (N_env,)]
        """
        total_reward = torch.zeros(
            (context.num_envs,),
            dtype=torch.float32,
            device=context.prev_positions.device,
        )
        weighted_terms: dict[str, torch.Tensor] = {}
        raw_terms: dict[str, torch.Tensor] = {}

        for term in self.terms:
            if not term.enabled:
                continue

            raw_value = torch.as_tensor(
                term.fn(context, term.params),
                dtype=torch.float32,
                device=context.prev_positions.device,
            )
            if tuple(raw_value.shape) != (context.num_envs,):
                raise ValueError(
                    f"Reward term {term.name} must return shape (N_env,), got {raw_value.shape}"
                )

            weighted_value = float(term.weight) * raw_value
            total_reward = total_reward + weighted_value
            weighted_terms[term.name] = weighted_value
            raw_terms[term.name] = raw_value

        return total_reward, weighted_terms, raw_terms


def build_reward_manager(config: RewardConfig) -> RewardManager:
    return RewardManager(terms=config.terms)


__all__ = [
    "RewardBatchContext",
    "RewardConfig",
    "RewardTermCfg",
    "RewardManager",
    "RewardTermFn",
    "build_reward_manager",
    "term_progress_to_goal",
    "term_collision_any",
    "term_success",
    "term_constant",
]
