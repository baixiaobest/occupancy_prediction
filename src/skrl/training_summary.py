from __future__ import annotations

import gymnasium as gym
import numpy as np


def _mean_metric(tracking_data: dict[str, list[float]], key: str) -> float | None:
    values = tracking_data.get(key)
    if not values:
        return None
    return float(np.mean(values))


def install_agent_tracking_summary(agent, *, prefix: str = "[train_skrl]") -> None:
    """Print rich PPO metrics whenever SKRL writes tracked data."""

    original_write_tracking_data = agent.write_tracking_data
    last_seen: dict[str, float] = {}

    def _patched_write_tracking_data(timestep: int, timesteps: int) -> None:
        td = agent.tracking_data
        metric_keys = {
            "total_reward_mean": "Reward / Total reward (mean)",
            "episode_len_mean": "Episode / Total timesteps (mean)",
            "policy_loss": "Loss / Policy loss",
            "value_loss": "Loss / Value loss",
            "entropy_loss": "Loss / Entropy loss",
            "action_std": "Policy / Standard deviation",
            "learning_rate": "Learning / Learning rate",
            "clip_ratio": "Policy / Clip fraction",
            "approx_kl": "Policy / Approx KL",
        }

        current_values: dict[str, float | None] = {
            name: _mean_metric(td, key) for name, key in metric_keys.items()
        }

        for name, value in current_values.items():
            if value is not None:
                last_seen[name] = float(value)

        total_reward_mean = current_values["total_reward_mean"]
        episode_len_mean = current_values["episode_len_mean"]
        policy_loss = current_values["policy_loss"] if current_values["policy_loss"] is not None else last_seen.get("policy_loss")
        value_loss = current_values["value_loss"] if current_values["value_loss"] is not None else last_seen.get("value_loss")
        entropy_loss = current_values["entropy_loss"] if current_values["entropy_loss"] is not None else last_seen.get("entropy_loss")
        action_std = current_values["action_std"] if current_values["action_std"] is not None else last_seen.get("action_std")
        learning_rate = current_values["learning_rate"] if current_values["learning_rate"] is not None else last_seen.get("learning_rate")
        clip_ratio = current_values["clip_ratio"] if current_values["clip_ratio"] is not None else last_seen.get("clip_ratio")
        approx_kl = current_values["approx_kl"] if current_values["approx_kl"] is not None else last_seen.get("approx_kl")

        has_any_metric = any(
            value is not None
            for value in (
                total_reward_mean,
                episode_len_mean,
                policy_loss,
                value_loss,
                entropy_loss,
                action_std,
                learning_rate,
                clip_ratio,
                approx_kl,
            )
        )

        if not has_any_metric:
            original_write_tracking_data(timestep, timesteps)
            return

        parts = [f"{prefix} ppo", f"t={int(timestep)}/{int(timesteps)}"]

        def _maybe_add(name: str, value: float | None, pattern: str) -> None:
            if value is not None:
                parts.append(f"{name}={pattern % value}\n")

        _maybe_add("ret", total_reward_mean, "%.3f")
        _maybe_add("ep_len", episode_len_mean, "%.1f")
        _maybe_add("pi_loss", policy_loss, "%.5f")
        _maybe_add("vf_loss", value_loss, "%.5f")
        _maybe_add("ent_loss", entropy_loss, "%.5f")
        _maybe_add("act_std", action_std, "%.5f")
        _maybe_add("lr", learning_rate, "%.6f")
        _maybe_add("clip_frac", clip_ratio, "%.5f")
        _maybe_add("approx_kl", approx_kl, "%.6f")

        print(" ".join(parts), flush=True)

        original_write_tracking_data(timestep, timesteps)

    agent.write_tracking_data = _patched_write_tracking_data


class PeriodicEpisodeSummaryWrapper(gym.Wrapper):
    """Print periodic episode summaries to the terminal during training."""

    def __init__(
        self,
        env: gym.Env,
        *,
        interval_episodes: int = 10,
        prefix: str = "[train_skrl]",
    ) -> None:
        super().__init__(env)
        interval = int(interval_episodes)
        if interval <= 0:
            raise ValueError("interval_episodes must be > 0")

        self.interval_episodes = interval
        self.prefix = str(prefix)

        self._global_steps = 0
        self._global_episodes = 0

        self._episode_return = 0.0
        self._episode_length = 0

        self._window_episode_count = 0
        self._window_return_sum = 0.0
        self._window_length_sum = 0
        self._window_success_count = 0
        self._window_collision_count = 0
        self._window_timeout_count = 0
        self._window_reward_term_sums: dict[str, float] = {}
        self._episode_reward_term_sums: dict[str, float] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_reward_term_sums.clear()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward_terms = info.get("reward_terms") if isinstance(info, dict) else None
        if isinstance(reward_terms, dict):
            for key, value in reward_terms.items():
                try:
                    term_value = float(value)
                except (TypeError, ValueError):
                    continue
                term_name = str(key)
                self._episode_reward_term_sums[term_name] = self._episode_reward_term_sums.get(term_name, 0.0) + term_value

        self._global_steps += 1
        self._episode_return += float(reward)
        self._episode_length += 1

        done = bool(terminated or truncated)
        if done:
            self._global_episodes += 1

            self._window_episode_count += 1
            self._window_return_sum += float(self._episode_return)
            self._window_length_sum += int(self._episode_length)
            self._window_success_count += int(bool(info.get("success", False)))
            self._window_collision_count += int(bool(info.get("collision", False)))
            self._window_timeout_count += int(bool(info.get("timeout", False)))

            for key, value in self._episode_reward_term_sums.items():
                self._window_reward_term_sums[key] = self._window_reward_term_sums.get(key, 0.0) + float(value)

            self._episode_return = 0.0
            self._episode_length = 0
            self._episode_reward_term_sums.clear()

            if self._window_episode_count >= self.interval_episodes:
                n = float(self._window_episode_count)
                mean_return = self._window_return_sum / n
                mean_len = float(self._window_length_sum) / n
                success_rate = float(self._window_success_count) / n
                collision_rate = float(self._window_collision_count) / n
                timeout_rate = float(self._window_timeout_count) / n

                reward_breakdown = ""
                if self._window_reward_term_sums:
                    parts: list[str] = []
                    for key in sorted(self._window_reward_term_sums):
                        mean_value = self._window_reward_term_sums[key] / n
                        parts.append(f"{key}={mean_value:.4f}")
                    if parts:
                        reward_breakdown = "reward_terms_mean_per_episode\n" + "\n".join(parts)

                print(
                    f"{self.prefix} summary \n"
                    f"episodes={self._global_episodes - self._window_episode_count + 1}-{self._global_episodes} \n"
                    f"steps={self._global_steps} \n"
                    f"mean_return={mean_return:.4f} \n"
                    f"mean_len={mean_len:.1f} \n"
                    f"success_rate={success_rate:.3f} \n"
                    f"collision_rate={collision_rate:.3f} \n"
                    f"timeout_rate={timeout_rate:.3f} \n"
                    f"{reward_breakdown}\n",
                    flush=True,
                )

                self._window_episode_count = 0
                self._window_return_sum = 0.0
                self._window_length_sum = 0
                self._window_success_count = 0
                self._window_collision_count = 0
                self._window_timeout_count = 0
                self._window_reward_term_sums.clear()

        return observation, reward, terminated, truncated, info


__all__ = ["PeriodicEpisodeSummaryWrapper", "install_agent_tracking_summary"]