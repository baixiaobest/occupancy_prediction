"""RL training utilities and building blocks."""

from .counterfactual import (
	CounterfactualRolloutBatch,
	integrate_velocity_plans,
	rollout_counterfactual_futures,
	sample_random_velocity_plans,
)
from .env_single import ORCASimConfig, ORCASingleEnv, RewardConfig, SingleEnvConfig
from .reward_manager import (
	RewardBatchContext,
	RewardManager,
	RewardTermCfg,
	RewardTermFn,
	term_collision_any,
	term_constant,
	term_progress_to_goal,
	term_success,
)
from .replay_buffer import ReplayBuffer, ReplaySampleBatch, TensorDict

__all__ = [
	"CounterfactualRolloutBatch",
	"ORCASimConfig",
	"ORCASingleEnv",
	"RewardConfig",
	"RewardBatchContext",
	"RewardManager",
	"RewardTermCfg",
	"RewardTermFn",
	"SingleEnvConfig",
	"ReplayBuffer",
	"ReplaySampleBatch",
	"TensorDict",
	"integrate_velocity_plans",
	"rollout_counterfactual_futures",
	"sample_random_velocity_plans",
	"term_collision_any",
	"term_constant",
	"term_progress_to_goal",
	"term_success",
]
