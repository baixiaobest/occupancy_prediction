# rl package guide

This package contains reusable RL building blocks for random-candidate action selection and Q-learning over simulated futures.

## What changed in the refactor

The package moved from a mostly flat file layout to a grouped layout by responsibility.

- Networks moved into `networks/`.
- Collectors moved into `collectors/`.
- Environment wrappers moved into `envs/`.
- Observation/reward managers moved into `managers/`.
- Trainers moved into `q_trainers/`.
- Training application and profiling moved into `training/`.

## Current package layout

```text
src/rl/
	__init__.py
	README.md
	counterfactual.py
	replay_buffer.py
	collectors/
		collector_base.py
		collector.py
		simple_collector.py
	envs/
		env_single.py
	managers/
		observation_manager.py
		reward_manager.py
	networks/
		proposal_network.py
		q_common.py
		q_network.py
		simple_q_network.py
	q_trainers/
		q_trainer_base.py
		q_trainer.py
		simple_q_trainer.py
	training/
		training_app.py
		training_profiler.py
```

## Module map

- `counterfactual.py`: Candidate velocity-plan sampling and future rollout integration.
- `replay_buffer.py`: Replay storage and batch sampling utilities.
- `collectors/`: Data collection policy logic and rollout-to-buffer transfer.
- `envs/`: ORCA single-environment wrappers.
- `managers/`: Observation terms and reward terms/configuration.
- `networks/`: Q networks, proposal networks, and shared Q math helpers.
- `q_trainers/`: Training-step logic and shared trainer base.
- `training/`: End-to-end training app orchestration and profiling utilities.
- `__init__.py`: Consolidated package exports used by scripts.

## Import guide

Use package exports for commonly used runtime components:

```python
from src.rl import ORCASingleEnv, RandomPlanCollector, RandomCandidateQTrainer
```

Import network builders from `networks` directly:

```python
from src.rl.networks.q_network import build_q_network
from src.rl.networks.simple_q_network import build_simple_q_network
from src.rl.networks.proposal_network import build_proposal_network
```

Import training app/profiler from `training` directly:

```python
from src.rl.training.training_app import RLTrainingApp
from src.rl.training.training_profiler import RunProfiler
```

## Typical runtime flow

1. Create `ORCASingleEnv` with reward and observation configs.
2. Use a collector (`RandomPlanCollector` or `SimpleRandomActionCollector`) to gather transitions.
3. Push transitions into `ReplayBuffer`.
4. Run trainer step (`RandomCandidateQTrainer` or `SimpleRandomCandidateQTrainer`) on sampled batches.
5. Periodically soft-update target networks and log training stats/checkpoints.

## Suggested reading order for AI agents

1. `__init__.py` for public API surface.
2. `envs/env_single.py`, then `managers/observation_manager.py` and `managers/reward_manager.py`.
3. `collectors/collector.py` (or `collectors/simple_collector.py`) plus `replay_buffer.py`.
4. `networks/q_network.py` and `networks/simple_q_network.py`.
5. `q_trainers/q_trainer.py` (or `q_trainers/simple_q_trainer.py`).
6. `training/training_app.py` for orchestration.