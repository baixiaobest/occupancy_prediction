from __future__ import annotations

import os

import pytest
import torch

from src.rl.reward_manager import RewardBatchContext, RewardConfig, RewardManager, RewardTermCfg, build_reward_manager


def _make_reward_context() -> RewardBatchContext:
    prev_positions = torch.tensor(
        [
            [[0.0, 0.0], [1.5, 0.0]],
            [[0.5, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    new_positions = torch.tensor(
        [
            [[0.5, 0.0], [0.75, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    goals = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    return RewardBatchContext(
        prev_positions=prev_positions,
        new_positions=new_positions,
        goals=goals,
        controlled_agent_indices=torch.tensor([0, 0], dtype=torch.int64),
        goal_tolerances=torch.tensor([0.2, 0.2], dtype=torch.float32),
    )


def test_build_reward_manager_default_config_outputs_expected_terms() -> None:
    manager = build_reward_manager(RewardConfig())
    total_reward, weighted_terms, raw_terms = manager.compute(_make_reward_context())

    assert set(weighted_terms.keys()) == {"progress", "step_penalty", "collision", "success"}
    assert set(raw_terms.keys()) == {"progress", "step_penalty", "collision", "success"}

    assert torch.allclose(raw_terms["progress"], torch.tensor([0.5, 0.5], dtype=torch.float32))
    assert torch.allclose(raw_terms["step_penalty"], torch.tensor([1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(raw_terms["collision"], torch.tensor([1.0, 0.0], dtype=torch.float32))
    assert torch.allclose(raw_terms["success"], torch.tensor([0.0, 1.0], dtype=torch.float32))

    assert torch.allclose(weighted_terms["step_penalty"], torch.zeros(2, dtype=torch.float32))
    assert torch.allclose(total_reward, torch.tensor([-0.5, 5.5], dtype=torch.float32))


def test_reward_manager_raises_on_invalid_term_shape() -> None:
    manager = RewardManager(
        terms=[
            RewardTermCfg(
                name="bad",
                fn=lambda context, params: torch.zeros((context.num_envs, 1), dtype=torch.float32),
            )
        ]
    )

    with pytest.raises(ValueError, match=r"must return shape \(N_env,\)"):
        manager.compute(_make_reward_context())


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    raise SystemExit(pytest.main([__file__]))