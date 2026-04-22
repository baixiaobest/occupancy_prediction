# src package guide

This folder contains the core data pipeline, models, simulation utilities, and RL components for occupancy prediction and action-value training.

## High-level flow

1. Define scene layouts and templates.
2. Simulate trajectories and build rollout occupancy tensors.
3. Build train/val datasets from rollout files.
4. Train prediction and Q-value models.
5. Evaluate and visualize rollouts and model behavior.

## Module map

- `scene.py`: Core scene data structures (agents, obstacles, paths, regions).
- `scene_template.py`: Parametric scene generators (corridor variants and base template API).
- `templates.py`: Ready-to-use template factories.
- `ORCASim.py`: ORCA-based simulator wrapper used to roll scenes forward.
- `rollout_setting.py`: Rollout settings container.
- `rollout_data.py`: Rollout payload dataclasses.
- `rollout_helpers.py`: Main rollout assembly and serialization helpers.
- `rollout_visualization.py`: Animation/visualization utilities for rollout tensors.
- `occupancy2d.py`: 2D occupancy rasterization utilities.
- `occupancy_patch.py`: Local centered occupancy patch extraction.
- `Dataset.py`: Dataset classes and dataset-building logic for occupancy windows.
- `network_common.py`: Shared model blocks (downsample/upsample and video reshape helpers).
- `VAE_prediction.py`: Occupancy prediction VAE encoder/decoder definitions.
- `proposal_network.py`: Velocity proposal models used for candidate action generation.
- `q_network.py`: Q-network over velocity-plan candidates.
- `simple_q_network.py`: Lightweight Q-network for reduced state inputs.
- `loss.py`: Reconstruction and KL-style losses.
- `rl/`: Reinforcement-learning package (collectors, env, replay, reward, trainers).
- `__init__.py`: Package exports for top-level symbols.

## Where scripts connect

- `scripts/ORCA_rollout.py` uses scene/templates + rollout helpers to generate data.
- `scripts/train_VAE.py` uses dataset + VAE modules + losses.
- `scripts/train_rl.py` uses `src/rl` trainers, env wrappers, and Q networks.
- `scripts/visualize_*.py` uses rollout and model visualization helpers.

## Suggested reading order for AI agents

1. `scene_template.py` and `templates.py` to understand scenario generation.
2. `rollout_helpers.py` and `rollout_data.py` to understand data format creation.
3. `Dataset.py` to understand model-ready tensor shapes.
4. `VAE_prediction.py`, `proposal_network.py`, `q_network.py` for learning models.
5. `rl/README.md` and `src/rl/*.py` for RL collection/training mechanics.

## Notes

- Ignore `__pycache__/` folders; they are generated artifacts.
- Most tensor-heavy code uses PyTorch and assumes batched inputs.