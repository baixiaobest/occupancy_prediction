# rl package guide

This package contains reusable RL building blocks for random-candidate action selection and Q-learning over simulated futures.

## Core responsibilities

- Build per-step observations from simulator state.
- Define reward terms and aggregate reward signals.
- Sample counterfactual candidate action plans.
- Collect transitions into replay buffers.
- Train Q-models from replayed batches.

## Module map

- `env_single.py`: Single-environment wrapper around ORCA simulation and managers.
- `observation_manager.py`: Observation term system and occupancy-context feature builders.
- `reward_manager.py`: Reward term system and default reward configuration.
- `counterfactual.py`: Candidate velocity-plan sampling and future rollout integration.
- `collector_base.py`: Abstract collector interface.
- `collector.py`: Main collector for velocity-plan candidate RL.
- `simple_collector.py`: Collector for simple-state RL variant.
- `replay_buffer.py`: Replay storage and batch sampling utilities.
- `q_common.py`: Shared Q-learning math helpers (soft update, TD target, sampling).
- `q_trainer_base.py`: Shared trainer validation and abstract trainer surface.
- `q_trainer.py`: Trainer for candidate-based Q-network pipeline.
- `simple_q_trainer.py`: Trainer for simple-state Q-network pipeline.
- `__init__.py`: Consolidated exports for package-level imports.

## Typical runtime flow

1. Create `ORCASingleEnv` with reward and observation configs.
2. Use a collector (`RandomPlanCollector` or `SimpleRandomActionCollector`) to gather transitions.
3. Push transitions into `ReplayBuffer`.
4. Run trainer step (`RandomCandidateQTrainer` or `SimpleRandomCandidateQTrainer`) on sampled batches.
5. Periodically soft-update target networks and log training stats.

## Key extension points

- Add custom observation terms in `observation_manager.py`.
- Add custom reward terms in `reward_manager.py`.
- Swap candidate samplers/rollout functions in trainer configs.
- Replace Q-network implementations while keeping trainer interfaces.

## Suggested entry points for AI agents

1. Start with `__init__.py` for exported API surface.
2. Read `env_single.py`, `observation_manager.py`, and `reward_manager.py`.
3. Read `collector.py` and `replay_buffer.py` to follow data collection.
4. Read `q_trainer.py` (or `simple_q_trainer.py`) for learning updates.