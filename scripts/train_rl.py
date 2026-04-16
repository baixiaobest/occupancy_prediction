from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualize_model import build_models
from src.q_network import build_q_network
from src.rl import (
    QActionSelectionConfig,
    ORCASimConfig,
    ORCASingleEnv,
    QTrainerConfig,
    RandomCandidateQTrainer,
    RandomPlanCollector,
    RandomPlanCollectorConfig,
    ReplayBuffer,
    RewardConfig,
    SingleEnvConfig,
    build_online_occupancy_observation_config,
)
from src.rl.counterfactual import rollout_counterfactual_futures, sample_random_velocity_plans
from src.rl.observation_manager import ObservationBatchContext, build_observation_manager, OnlineOccupancyObservationConfig
from src.scene import Scene
from src.templates import cross_templates, default_templates, l_shape_templates, test_templates


@dataclass
class EvaluationSummary:
    episodes: int
    total_reward: float
    mean_reward: float
    mean_episode_length: float
    success_rate: float
    collision_rate: float
    timeout_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL selection/Q network with ORCA scenes")
    parser.add_argument("--decoder-checkpoint", type=Path, required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--template-set", choices=["default", "test", "cross", "l_shape"], default="default")
    parser.add_argument("--scene-selection", choices=["random", "cycle", "fixed"], default="random")
    parser.add_argument("--fixed-scene-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--collect-steps-per-iter", type=int, default=8)
    parser.add_argument("--updates-per-iter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--num-bootstrap-candidates", type=int, default=8)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-max-speed", type=float, default=2.0)
    parser.add_argument("--candidate-delta-std", type=float, default=0.25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--loss-type", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--tap-layer", type=int, default=1)
    parser.add_argument("--occupancy-resolution", type=float, default=0.1)
    parser.add_argument("--env-max-steps", type=int, default=200)
    parser.add_argument("--controlled-agent-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/rl_q_selection.pt"))
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-scene-selection", choices=["random", "cycle", "fixed"], default="cycle")
    parser.add_argument("--eval-seed-offset", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _select_templates(template_set: str):
    if template_set == "default":
        return default_templates()
    if template_set == "test":
        return test_templates()
    if template_set == "cross":
        return cross_templates()
    if template_set == "l_shape":
        return l_shape_templates()
    raise ValueError(f"Unknown template set: {template_set}")


def _build_scene_pool(template_set: str) -> list[Scene]:
    scenes: list[Scene] = []
    for template in _select_templates(template_set):
        scenes.extend(template.generate())
    if not scenes:
        raise ValueError(f"No scenes generated for template set {template_set}")
    return scenes


def _make_scene_factory(
    scenes: list[Scene],
    *,
    selection: str,
    fixed_scene_index: int,
    seed: int,
):
    rng = random.Random(int(seed))
    scene_count = len(scenes)
    if scene_count == 0:
        raise ValueError("scenes must not be empty")
    fixed_idx = int(fixed_scene_index) % scene_count
    next_idx = 0

    def factory() -> Scene:
        nonlocal next_idx
        if selection == "fixed":
            scene = scenes[fixed_idx]
        elif selection == "cycle":
            scene = scenes[next_idx % scene_count]
            next_idx += 1
        else:
            scene = scenes[rng.randrange(scene_count)]
        return copy.deepcopy(scene)

    return factory


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint_model_config(checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise ValueError("decoder checkpoint must contain a 'model_config' dictionary")
    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("decoder checkpoint model_config must be a dict")
    return model_config


def _probe_tap_channels(
    *,
    env: ORCASingleEnv,
    observation_config,
    decoder: torch.nn.Module,
    horizon: int,
    max_speed: float,
    delta_std: float,
    dt: float,
    tap_layer: int,
    latent_channels: int,
    latent_shape: tuple[int, int, int],
) -> int:
    observation_manager = build_observation_manager(observation_config)
    raw_obs = env.reset(seed=0)
    scene = env.sim.scene if env.sim is not None else None
    if scene is None:
        raise RuntimeError("Probe reset did not initialize env.sim.scene")
    obs = observation_manager.compute(ObservationBatchContext(raw_obs=raw_obs, scene=scene))
    probe_candidates = sample_random_velocity_plans(
        current_velocity=obs["current_velocity"],
        num_candidates=1,
        horizon=int(horizon),
        max_speed=float(max_speed),
        delta_std=float(delta_std),
        dt=float(dt),
        include_current_velocity_candidate=True,
    )
    rollout = rollout_counterfactual_futures(
        decoder=decoder,
        dynamic_context=obs["dynamic_context"],
        static_map=obs["static_map"],
        candidate_velocity_plans=probe_candidates,
        latent_channels=int(latent_channels),
        latent_shape=latent_shape,
        dt=float(dt),
        tap_layer=int(tap_layer),
    )
    if rollout.tapped_features is None:
        raise RuntimeError("Decoder rollout did not return tapped features for the selected tap layer")
    return int(rollout.tapped_features.shape[3])


def _build_checkpoint(
    *,
    iteration: int,
    args: argparse.Namespace,
    q_network: torch.nn.Module,
    target_q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    q_model_config: dict[str, object],
    trainer_config: QTrainerConfig,
    collector_config: RandomPlanCollectorConfig,
    env_config: SingleEnvConfig,
) -> dict[str, object]:
    return {
        "iteration": int(iteration),
        "args": vars(args),
        "q_network": q_network.state_dict(),
        "target_q_network": target_q_network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "q_model_config": q_model_config,
        "trainer_config": {
            "discount": trainer_config.discount,
            "target_tau": trainer_config.target_tau,
            "selection_temperature": trainer_config.selection_temperature,
            "selection_seed": trainer_config.selection_seed,
            "num_bootstrap_candidates": trainer_config.num_bootstrap_candidates,
            "max_speed": trainer_config.max_speed,
            "delta_std": trainer_config.delta_std,
            "dt": trainer_config.dt,
            "tap_layer": trainer_config.tap_layer,
            "latent_channels": trainer_config.latent_channels,
            "latent_shape": trainer_config.latent_shape,
            "include_current_velocity_candidate": trainer_config.include_current_velocity_candidate,
            "binary_feedback": trainer_config.binary_feedback,
            "threshold": trainer_config.threshold,
            "grad_clip_norm": trainer_config.grad_clip_norm,
            "loss_type": trainer_config.loss_type,
        },
        "collector_config": {
            "horizon": collector_config.horizon,
            "num_candidates": collector_config.num_candidates,
            "max_speed": collector_config.max_speed,
            "delta_std": collector_config.delta_std,
            "dt": collector_config.dt,
            "include_current_velocity_candidate": collector_config.include_current_velocity_candidate,
            "action_selection": collector_config.action_selection,
            "seed": collector_config.seed,
        },
        "env_config": {
            "max_steps": env_config.max_steps,
            "controlled_agent_index": env_config.controlled_agent_index,
            "device": env_config.device,
        },
    }


def _log(message: str) -> None:
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    stream.write(f"{message}\n")
    stream.flush()


class RLTrainingApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.output_path = args.output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.decoder: torch.nn.Module | None = None
        self.history_len: int | None = None
        self.decoder_context_len: int | None = None
        self.latent_shape: tuple[int, int, int] | None = None
        self.latent_channels: int | None = None
        self.local_map_shape: tuple[int, int] | None = None
        self.observation_config = None
        self.env_config: SingleEnvConfig | None = None
        self.sim_config: ORCASimConfig | None = None
        self.scenes: list[Scene] = []
        self.env: ORCASingleEnv | None = None
        self.tap_channels: int | None = None
        self.q_model_config: dict[str, object] | None = None
        self.q_network: torch.nn.Module | None = None
        self.target_q_network: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.collector_config: RandomPlanCollectorConfig | None = None
        self.collector: RandomPlanCollector | None = None
        self.trainer_config: QTrainerConfig | None = None
        self.trainer: RandomCandidateQTrainer | None = None
        self.start_time: float | None = None

    def run(self) -> None:
        _seed_everything(int(self.args.seed))
        self._setup_runtime()
        self._log_setup()
        self.start_time = time.time()
        self._run_warmup()
        for iteration in range(1, int(self.args.iterations) + 1):
            self._run_iteration(iteration)
        self._save_final_checkpoint()

    def _setup_runtime(self) -> None:
        self._setup_decoder()
        self._setup_environment()
        self._setup_q_networks()
        self._setup_collector_and_trainer()

    def _setup_decoder(self) -> None:
        decoder, history_len, decoder_context_len, latent_shape, latent_channels = build_models(
            checkpoint_path=self.args.decoder_checkpoint,
            device=self.device,
        )
        for param in decoder.parameters():
            param.requires_grad_(False)
        decoder.eval()

        model_config = _load_checkpoint_model_config(self.args.decoder_checkpoint)
        input_shape = tuple(int(v) for v in model_config["input_shape"])
        if len(input_shape) != 4:
            raise ValueError(f"Expected checkpoint input_shape to have 4 dims, got {input_shape}")

        self.decoder = decoder
        self.history_len = int(history_len)
        self.decoder_context_len = int(decoder_context_len)
        self.latent_shape = latent_shape
        self.latent_channels = int(latent_channels)
        self.local_map_shape = (int(input_shape[2]), int(input_shape[3]))
        self.observation_config = build_online_occupancy_observation_config(
            OnlineOccupancyObservationConfig(
                decoder_context_len=self.decoder_context_len,
                local_map_shape=self.local_map_shape,
                occupancy_resolution=(float(self.args.occupancy_resolution), float(self.args.occupancy_resolution)),
                device=str(self.device),
            )
        )

    def _setup_environment(self) -> None:
        self.env_config = SingleEnvConfig(
            max_steps=int(self.args.env_max_steps),
            controlled_agent_index=int(self.args.controlled_agent_index),
            device=str(self.device),
            reward=RewardConfig(),
            observation=self.observation_config,
        )
        self.sim_config = ORCASimConfig(time_step=0.1)
        self.scenes = _build_scene_pool(self.args.template_set)
        scene_factory = _make_scene_factory(
            self.scenes,
            selection=self.args.scene_selection,
            fixed_scene_index=int(self.args.fixed_scene_index),
            seed=int(self.args.seed),
        )
        self.env = ORCASingleEnv(
            scene_factory=scene_factory,
            sim_config=self.sim_config,
            env_config=self.env_config,
        )

    def _setup_q_networks(self) -> None:
        if self.env is None or self.decoder is None or self.observation_config is None:
            raise RuntimeError("Environment, decoder, and observation config must be initialized before Q setup")
        if self.sim_config is None or self.latent_channels is None or self.latent_shape is None:
            raise RuntimeError("Simulation and latent metadata must be initialized before Q setup")

        self.tap_channels = _probe_tap_channels(
            env=self.env,
            observation_config=self.observation_config,
            decoder=self.decoder,
            horizon=int(self.args.horizon),
            max_speed=float(self.args.candidate_max_speed),
            delta_std=float(self.args.candidate_delta_std),
            dt=float(self.sim_config.time_step),
            tap_layer=int(self.args.tap_layer),
            latent_channels=self.latent_channels,
            latent_shape=self.latent_shape,
        )
        tapped_feature_channels = int(self.args.horizon) * self.tap_channels
        self.q_model_config = {
            "tapped_feature_channels": tapped_feature_channels,
            "horizon": int(self.args.horizon),
        }
        self.q_network = build_q_network(
            tapped_feature_channels=tapped_feature_channels,
            horizon=int(self.args.horizon),
            device=self.device,
        )
        self.target_q_network = build_q_network(
            tapped_feature_channels=tapped_feature_channels,
            horizon=int(self.args.horizon),
            device=self.device,
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        self.replay_buffer = ReplayBuffer(capacity=int(self.args.replay_capacity), seed=int(self.args.seed))

    def _setup_collector_and_trainer(self) -> None:
        if self.decoder is None or self.latent_channels is None or self.latent_shape is None:
            raise RuntimeError("Decoder and latent metadata must be initialized before collector setup")
        if self.q_network is None or self.target_q_network is None or self.optimizer is None:
            raise RuntimeError("Q networks and optimizer must be initialized before trainer setup")
        if self.env is None or self.replay_buffer is None or self.sim_config is None:
            raise RuntimeError("Environment, replay buffer, and simulation config must be initialized before trainer setup")

        q_selection_config = QActionSelectionConfig(
            temperature=float(self.args.selection_temperature),
            seed=int(self.args.seed),
            tap_layer=int(self.args.tap_layer),
            latent_channels=self.latent_channels,
            latent_shape=self.latent_shape,
        )
        self.collector_config = RandomPlanCollectorConfig(
            horizon=int(self.args.horizon),
            num_candidates=int(self.args.num_candidates),
            max_speed=float(self.args.candidate_max_speed),
            delta_std=float(self.args.candidate_delta_std),
            dt=float(self.sim_config.time_step),
            include_current_velocity_candidate=True,
            action_selection="q_softmax",
            seed=int(self.args.seed),
            q_selection=q_selection_config,
        )
        self.collector = RandomPlanCollector(
            env=self.env,
            replay_buffer=self.replay_buffer,
            observation_manager=None,
            config=self.collector_config,
            q_network=self.q_network,
            decoder=self.decoder,
        )
        self.trainer_config = QTrainerConfig(
            discount=float(self.args.discount),
            target_tau=float(self.args.target_tau),
            selection_temperature=float(self.args.selection_temperature),
            selection_seed=int(self.args.seed) + 1,
            num_bootstrap_candidates=int(self.args.num_bootstrap_candidates),
            max_speed=float(self.args.candidate_max_speed),
            delta_std=float(self.args.candidate_delta_std),
            dt=float(self.sim_config.time_step),
            tap_layer=int(self.args.tap_layer),
            latent_channels=self.latent_channels,
            latent_shape=self.latent_shape,
            include_current_velocity_candidate=True,
            grad_clip_norm=float(self.args.grad_clip_norm) if self.args.grad_clip_norm is not None else None,
            loss_type=str(self.args.loss_type),
        )
        self.trainer = RandomCandidateQTrainer(
            q_network=self.q_network,
            target_q_network=self.target_q_network,
            decoder=self.decoder,
            optimizer=self.optimizer,
            config=self.trainer_config,
        )

    def _log_setup(self) -> None:
        _log(
            "RL setup: "
            f"{len(self.scenes)} scenes | history_len={self.history_len} | decoder_context_len={self.decoder_context_len} | "
            f"local_map_shape={self.local_map_shape} | tap_channels={self.tap_channels} | device={self.device}"
        )

    def _run_warmup(self) -> None:
        if int(self.args.warmup_steps) <= 0:
            return
        if self.collector is None:
            raise RuntimeError("Collector must be initialized before warmup")
        warmup_summary = self.collector.collect_steps(int(self.args.warmup_steps), reset_seed=int(self.args.seed))
        _log(
            f"Warmup collected {warmup_summary.transitions_added} transitions | "
            f"episodes={warmup_summary.episodes_completed} | reward={warmup_summary.total_reward:.3f}"
        )

    def _run_iteration(self, iteration: int) -> None:
        if self.q_network is None or self.collector is None or self.replay_buffer is None or self.trainer is None:
            raise RuntimeError("Training runtime must be initialized before iterations")

        self.q_network.eval()
        collect_summary = self.collector.collect_steps(int(self.args.collect_steps_per_iter))

        last_stats = None
        if len(self.replay_buffer) >= int(self.args.batch_size):
            self.q_network.train()
            for _ in range(int(self.args.updates_per_iter)):
                batch = self.replay_buffer.sample(batch_size=int(self.args.batch_size), device=self.device)
                last_stats = self.trainer.train_step(batch)
        else:
            self.q_network.train()

        if iteration == 1 or iteration % int(self.args.log_interval) == 0:
            self._log_iteration(iteration, collect_summary, last_stats)

        if self._should_evaluate(iteration):
            eval_summary = self._evaluate_policy(iteration)
            self._log_evaluation(iteration, eval_summary)

        if iteration % int(self.args.save_interval) == 0:
            self._save_checkpoint(iteration)

    def _should_evaluate(self, iteration: int) -> bool:
        return (
            int(self.args.eval_interval) > 0
            and int(self.args.eval_episodes) > 0
            and iteration % int(self.args.eval_interval) == 0
        )

    def _evaluate_policy(self, iteration: int) -> EvaluationSummary:
        if self.env_config is None or self.sim_config is None or self.collector_config is None:
            raise RuntimeError("Environment and collector configs must be initialized before evaluation")
        if self.q_network is None or self.decoder is None:
            raise RuntimeError("Q network and decoder must be initialized before evaluation")

        eval_seed_base = int(self.args.seed) + int(self.args.eval_seed_offset)
        eval_scene_factory = _make_scene_factory(
            self.scenes,
            selection=str(self.args.eval_scene_selection),
            fixed_scene_index=int(self.args.fixed_scene_index),
            seed=eval_seed_base,
        )
        eval_env = ORCASingleEnv(
            scene_factory=eval_scene_factory,
            sim_config=self.sim_config,
            env_config=self.env_config,
        )
        eval_collector_config = copy.deepcopy(self.collector_config)
        eval_collector_config.seed = eval_seed_base
        if eval_collector_config.q_selection is not None:
            eval_collector_config.q_selection.seed = eval_seed_base + iteration
        eval_collector = RandomPlanCollector(
            env=eval_env,
            replay_buffer=ReplayBuffer(capacity=1, seed=eval_seed_base),
            observation_manager=None,
            config=eval_collector_config,
            q_network=self.q_network,
            decoder=self.decoder,
        )

        was_training = self.q_network.training
        self.q_network.eval()
        try:
            total_reward = 0.0
            total_steps = 0
            successes = 0
            collisions = 0
            timeouts = 0
            episodes = int(self.args.eval_episodes)
            for episode_idx in range(episodes):
                obs = eval_collector.reset_episode(seed=eval_seed_base + episode_idx)
                episode_done = False
                episode_info: dict[str, object] = {}
                while not episode_done:
                    action_selection = eval_collector.select_action(obs)
                    next_raw_obs, rewards, dones, infos = eval_env.step(action_selection.selected_plan[0, 0])
                    total_reward += float(torch.as_tensor(rewards, dtype=torch.float32).sum().item())
                    total_steps += 1
                    episode_info = infos[0]
                    episode_done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
                    if not episode_done:
                        obs = eval_collector.prepare_observation(next_raw_obs)

                successes += int(bool(episode_info.get("success", False)))
                collisions += int(bool(episode_info.get("collision", False)))
                timeouts += int(bool(episode_info.get("timeout", False)))
        finally:
            if was_training:
                self.q_network.train()

        episodes_float = float(self.args.eval_episodes)
        return EvaluationSummary(
            episodes=int(self.args.eval_episodes),
            total_reward=total_reward,
            mean_reward=total_reward / episodes_float,
            mean_episode_length=float(total_steps) / episodes_float,
            success_rate=float(successes) / episodes_float,
            collision_rate=float(collisions) / episodes_float,
            timeout_rate=float(timeouts) / episodes_float,
        )

    def _log_evaluation(self, iteration: int, summary: EvaluationSummary) -> None:
        _log(
            f"eval iter={iteration:05d} | episodes={summary.episodes:03d} | total_reward={summary.total_reward:.3f} | "
            f"mean_reward={summary.mean_reward:.3f} | mean_ep_len={summary.mean_episode_length:.1f} | "
            f"success={summary.success_rate:.3f} | collision={summary.collision_rate:.3f} | timeout={summary.timeout_rate:.3f}"
        )

    def _log_iteration(self, iteration: int, collect_summary, last_stats) -> None:
        if self.replay_buffer is None or self.start_time is None:
            raise RuntimeError("Replay buffer and timer must be initialized before logging")

        elapsed = time.time() - self.start_time
        if last_stats is None:
            _log(
                f"iter={iteration:05d} | replay={len(self.replay_buffer):06d} | "
                f"collect_reward={collect_summary.total_reward:.3f} | collect_episodes={collect_summary.episodes_completed} | "
                f"updates=0 | elapsed={elapsed:.1f}s"
            )
            return

        _log(
            f"iter={iteration:05d} | replay={len(self.replay_buffer):06d} | "
            f"collect_reward={collect_summary.total_reward:.3f} | collect_episodes={collect_summary.episodes_completed} | "
            f"loss={last_stats.loss:.5f} | q={last_stats.q_pred_mean:.3f} | target={last_stats.target_mean:.3f} | "
            f"next_q={last_stats.next_q_mean:.3f} | sel_entropy={last_stats.selection_entropy_mean:.3f} | "
            f"done_frac={last_stats.done_fraction:.3f} | elapsed={elapsed:.1f}s"
        )

    def _build_checkpoint(self, iteration: int) -> dict[str, object]:
        if self.q_network is None or self.target_q_network is None or self.optimizer is None:
            raise RuntimeError("Q networks and optimizer must be initialized before checkpointing")
        if self.q_model_config is None or self.trainer_config is None or self.collector_config is None:
            raise RuntimeError("Trainer and collector configs must be initialized before checkpointing")
        if self.env_config is None:
            raise RuntimeError("Environment config must be initialized before checkpointing")

        return _build_checkpoint(
            iteration=iteration,
            args=self.args,
            q_network=self.q_network,
            target_q_network=self.target_q_network,
            optimizer=self.optimizer,
            q_model_config=self.q_model_config,
            trainer_config=self.trainer_config,
            collector_config=self.collector_config,
            env_config=self.env_config,
        )

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint = self._build_checkpoint(iteration)
        periodic_path = self.output_path.with_name(f"{self.output_path.stem}_iter_{iteration:06d}{self.output_path.suffix}")
        torch.save(checkpoint, periodic_path)
        torch.save(checkpoint, self.output_path)
        _log(f"Checkpoint saved: {periodic_path}")

    def _save_final_checkpoint(self) -> None:
        checkpoint = self._build_checkpoint(int(self.args.iterations))
        torch.save(checkpoint, self.output_path)
        _log(f"Training finished. Final checkpoint saved: {self.output_path}")


def main() -> None:
    RLTrainingApp(parse_args()).run()


if __name__ == "__main__":
    main()