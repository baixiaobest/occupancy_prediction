"""RL training utilities and building blocks."""

from .collector import (
	CollectSummary,
	RandomPlanCollector,
	RandomPlanCollectorConfig,
)
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
from .observation_manager import (
	ObservationBatchContext,
	ObservationManager,
	ObservationTermCfg,
	ObservationTermFn,
	OnlineOccupancyObservationConfig,
	build_online_occupancy_observation_manager,
	term_controlled_current_velocity,
	term_controlled_goal_offset,
	term_dynamic_local_occupancy_context,
	term_static_local_occupancy,
)
from .replay_buffer import ReplayBuffer, ReplaySampleBatch, TensorDict

__all__ = [
	"CollectSummary",
	"CounterfactualRolloutBatch",
	"ObservationBatchContext",
	"ObservationManager",
	"ObservationTermCfg",
	"ObservationTermFn",
	"OnlineOccupancyObservationConfig",
	"ORCASimConfig",
	"ORCASingleEnv",
	"RandomPlanCollector",
	"RandomPlanCollectorConfig",
	"RewardConfig",
	"RewardBatchContext",
	"RewardManager",
	"RewardTermCfg",
	"RewardTermFn",
	"SingleEnvConfig",
	"ReplayBuffer",
	"ReplaySampleBatch",
	"TensorDict",
	"build_online_occupancy_observation_manager",
	"integrate_velocity_plans",
	"rollout_counterfactual_futures",
	"sample_random_velocity_plans",
	"term_controlled_current_velocity",
	"term_controlled_goal_offset",
	"term_collision_any",
	"term_constant",
	"term_dynamic_local_occupancy_context",
	"term_progress_to_goal",
	"term_static_local_occupancy",
	"term_success",
]
