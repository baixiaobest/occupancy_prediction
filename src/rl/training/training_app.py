from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

try:
    import wandb
except ImportError:
    wandb = None

from ...experiment_utils import EmptyGoalTemplateConfig, build_scene_pool, seed_everything
from ..networks.q_network import build_q_network
from ...scene import Scene
from ...scene_sampling import make_scene_factory
from ..networks.simple_q_network import build_simple_q_network
from ..networks.simple_proposal_network import build_simple_proposal_network
from ..collectors.collector import QActionSelectionConfig, RandomPlanCollector, RandomPlanCollectorConfig
from ..counterfactual import rollout_counterfactual_futures, sample_random_velocity_plans
from ..envs.env_single import ORCASimConfig, ORCASingleEnv, SingleEnvConfig
from ..managers.observation_manager import (
    ObservationBatchContext,
    OnlineOccupancyObservationConfig,
    build_observation_manager,
    build_online_occupancy_observation_config,
    build_simple_state_observation_config,
)
from ..q_trainers.q_trainer import QTrainerConfig, RandomCandidateQTrainer
from ..replay_buffer import ReplayBuffer
from ..managers.reward_manager import RewardConfig
from ..collectors.simple_collector import (
    SimpleActionCollector,
    SimpleActionCollectorConfig,
    SimpleQActionSelectionConfig,
)
from ..q_trainers.simple_q_trainer import SimpleActionCandidateQTrainer, SimpleQTrainerConfig
from ...training_profiler import RunProfiler

DecoderBuilder = Callable[[Path, torch.device], tuple[torch.nn.Module, int, int, tuple[int, int, int], int]]


@dataclass
class EvaluationSummary:
    episodes: int
    total_reward: float
    mean_reward: float
    mean_episode_length: float
    success_rate: float
    collision_rate: float
    timeout_rate: float


def _log(message: str) -> None:
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    stream.write(f"{message}\n")
    stream.flush()


def _log_message(message: str, wandb_run: object | None = None) -> None:
    _log(message)
    if wandb_run is not None and wandb is not None:
        wandb.termlog(message)


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


class RLTrainingApp:
    def __init__(
        self,
        args: argparse.Namespace,
        profiler: RunProfiler | None = None,
        decoder_builder: DecoderBuilder | None = None,
    ) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.output_path = args.output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.decoder_builder = decoder_builder
        self.profiler = profiler or RunProfiler(enabled=False, top_n=0, output_path=None)
        self.wandb_run: object | None = self._init_wandb()
        self.profiler.set_logger(self.log)

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
        self.simple_proposal_network: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.collector_config: RandomPlanCollectorConfig | SimpleActionCollectorConfig | None = None
        self.collector: RandomPlanCollector | SimpleActionCollector | None = None
        self.trainer_config: QTrainerConfig | SimpleQTrainerConfig | None = None
        self.trainer: RandomCandidateQTrainer | SimpleActionCandidateQTrainer | None = None
        self._q_updates_since_target: int = 0
        self.start_time: float | None = None

    def _init_wandb(self) -> object | None:
        if not self.args.wandb:
            return None
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        return wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.wandb_run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            config=vars(self.args),
        )

    def log(self, message: str) -> None:
        _log_message(message, wandb_run=self.wandb_run)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def run(self) -> None:
        self.profiler.start()
        try:
            seed_everything(int(self.args.seed), include_numpy=False)
            with self.profiler.section("setup_runtime"):
                self._setup_runtime()
            self._log_setup()
            self.start_time = time.time()
            with self.profiler.section("warmup"):
                self._run_warmup()
            for iteration in range(1, int(self.args.iterations) + 1):
                self._run_iteration(iteration)
            with self.profiler.section("save_final_checkpoint"):
                self._save_final_checkpoint()
        finally:
            self.profiler.stop()
            self.profiler.report()

    def _setup_runtime(self) -> None:
        if int(self.args.target_update_interval) <= 0:
            raise ValueError("--target-update-interval must be > 0")
        if self.args.mode == "simple_state_q":
            with self.profiler.section("setup_environment"):
                self._setup_simple_environment()
            with self.profiler.section("setup_q_networks"):
                self._setup_simple_q_components()
            return
        with self.profiler.section("setup_decoder"):
            self._setup_decoder()
        with self.profiler.section("setup_environment"):
            self._setup_environment()
        with self.profiler.section("setup_q_networks"):
            self._setup_q_networks()
        with self.profiler.section("setup_collector_and_trainer"):
            self._setup_collector_and_trainer()

    def _setup_decoder(self) -> None:
        if self.args.decoder_checkpoint is None:
            raise ValueError("--decoder-checkpoint is required when mode='counterfactual_q'")
        if self.decoder_builder is None:
            raise ValueError("decoder_builder is required when mode='counterfactual_q'")

        decoder, history_len, decoder_context_len, latent_shape, latent_channels = self.decoder_builder(
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

    def _setup_simple_environment(self) -> None:
        self.observation_config = build_simple_state_observation_config()
        self.env_config = SingleEnvConfig(
            max_steps=int(self.args.env_max_steps),
            controlled_agent_index=int(self.args.controlled_agent_index),
            device=str(self.device),
            reward=RewardConfig(),
            observation=self.observation_config,
        )
        self.sim_config = ORCASimConfig(time_step=0.1)
        goal_distance_range = tuple(float(value) for value in self.args.empty_goal_distance_range)
        self.scenes = build_scene_pool(
            "empty_goal",
            empty_goal=EmptyGoalTemplateConfig(
                goal_distance_range=(goal_distance_range[0], goal_distance_range[1]),
                goal_seed=int(self.args.seed),
            ),
        )
        scene_factory = make_scene_factory(
            self.scenes,
            selection=str(self.args.scene_selection),
            fixed_scene_index=int(self.args.fixed_scene_index),
            seed=int(self.args.seed),
        )
        self.env = ORCASingleEnv(
            scene_factory=scene_factory,
            sim_config=self.sim_config,
            env_config=self.env_config,
        )

    def _setup_simple_q_components(self) -> None:
        if self.env is None or self.sim_config is None:
            raise RuntimeError("Environment and simulation config must be initialized before simple Q setup")

        self.q_model_config = {
            "type": "simple_state_q",
            "hidden_dims": [int(value) for value in self.args.simple_q_hidden_dims],
            "proposal_hidden_dims": [int(value) for value in self.args.simple_proposal_hidden_dims],
            "proposal_min_variance": float(self.args.simple_proposal_min_variance),
        }
        self.q_network = build_simple_q_network(
            hidden_dims=self.args.simple_q_hidden_dims,
            device=self.device,
        )
        self.target_q_network = build_simple_q_network(
            hidden_dims=self.args.simple_q_hidden_dims,
            device=self.device,
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        self.simple_proposal_network = build_simple_proposal_network(
            horizon=1,
            hidden_dims=self.args.simple_proposal_hidden_dims,
            min_variance=float(self.args.simple_proposal_min_variance),
            device=self.device,
        )
        self.replay_buffer = ReplayBuffer(capacity=int(self.args.replay_capacity), seed=int(self.args.seed))
        q_selection_config = SimpleQActionSelectionConfig(
            temperature=float(self.args.selection_temperature),
            seed=int(self.args.seed),
        )
        self.collector_config = SimpleActionCollectorConfig(
            num_candidates=int(self.args.num_candidates),
            include_current_velocity_candidate=True,
            action_selection="q_softmax",
            seed=int(self.args.seed),
            q_selection=q_selection_config,
            max_speed=float(self.args.candidate_max_speed),
        )
        self.collector = SimpleActionCollector(
            env=self.env,
            replay_buffer=self.replay_buffer,
            observation_manager=None,
            config=self.collector_config,
            proposal_network=self.simple_proposal_network,
            q_network=self.q_network,
        )
        self.trainer_config = SimpleQTrainerConfig(
            discount=float(self.args.discount),
            target_tau=float(self.args.target_tau),
            selection_temperature=float(self.args.selection_temperature),
            selection_seed=int(self.args.seed) + 1,
            num_bootstrap_candidates=int(self.args.num_bootstrap_candidates),
            max_speed=float(self.args.candidate_max_speed),
            delta_std=float(self.args.candidate_delta_std),
            dt=float(self.sim_config.time_step),
            include_current_velocity_candidate=True,
            grad_clip_norm=float(self.args.grad_clip_norm) if self.args.grad_clip_norm is not None else None,
            loss_type=str(self.args.loss_type),
        )
        self.trainer = SimpleActionCandidateQTrainer(
            q_network=self.q_network,
            target_q_network=self.target_q_network,
            proposal_network=self.simple_proposal_network,
            optimizer=self.optimizer,
            config=self.trainer_config,
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
        self.scenes = build_scene_pool(str(self.args.template_set))
        scene_factory = make_scene_factory(
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
        if self.args.mode == "simple_state_q":
            self.log(
                "RL setup: "
                f"mode={self.args.mode} | scenes={len(self.scenes)} | "
                f"goal_distance_range={tuple(float(v) for v in self.args.empty_goal_distance_range)} | "
                f"q_hidden_dims={list(self.args.simple_q_hidden_dims)} | "
                f"proposal_hidden_dims={list(self.args.simple_proposal_hidden_dims)} | "
                f"target_update_interval={int(self.args.target_update_interval)} | device={self.device}"
            )
            return
        self.log(
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
        self.log(
            f"Warmup collected {warmup_summary.transitions_added} transitions | "
            f"episodes={warmup_summary.episodes_completed} | reward={warmup_summary.total_reward:.3f}"
        )

    def _run_iteration(self, iteration: int) -> None:
        if self.q_network is None or self.collector is None or self.replay_buffer is None or self.trainer is None:
            raise RuntimeError("Training runtime must be initialized before iterations")

        self.q_network.eval()
        with self.profiler.section("collect_steps"):
            collect_summary = self.collector.collect_steps(int(self.args.collect_steps_per_iter))

        last_stats = None
        if len(self.replay_buffer) >= int(self.args.batch_size):
            self.q_network.train()
            with self.profiler.section("train_updates"):
                for _ in range(int(self.args.updates_per_iter)):
                    batch = self.replay_buffer.sample(batch_size=int(self.args.batch_size), device=self.device)
                    last_stats = self.trainer.train_step(batch)
                    self._q_updates_since_target += 1
                    if self._q_updates_since_target >= int(self.args.target_update_interval):
                        self.trainer.update_target_network()
                        self._q_updates_since_target = 0
        else:
            self.q_network.train()

        if iteration == 1 or iteration % int(self.args.log_interval) == 0:
            self._log_iteration(iteration, collect_summary, last_stats)

        eval_summary = None
        if self._should_evaluate(iteration):
            with self.profiler.section("evaluation"):
                eval_summary = self._evaluate_policy(iteration)
            self._log_evaluation(iteration, eval_summary)

        if iteration % int(self.args.save_interval) == 0:
            with self.profiler.section("save_checkpoint"):
                self._save_checkpoint(iteration)

        self._log_wandb_iteration(iteration, collect_summary, last_stats, eval_summary)

    def _should_evaluate(self, iteration: int) -> bool:
        return (
            int(self.args.eval_interval) > 0
            and int(self.args.eval_episodes) > 0
            and iteration % int(self.args.eval_interval) == 0
        )

    def _evaluate_policy(self, iteration: int) -> EvaluationSummary:
        if self.env_config is None or self.sim_config is None or self.collector_config is None:
            raise RuntimeError("Environment and collector configs must be initialized before evaluation")
        if self.q_network is None:
            raise RuntimeError("Q network must be initialized before evaluation")

        eval_seed_base = int(self.args.seed) + int(self.args.eval_seed_offset)
        eval_scene_factory = make_scene_factory(
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
        if getattr(eval_collector_config, "q_selection", None) is not None:
            eval_collector_config.q_selection.seed = eval_seed_base + iteration
        if self.args.mode == "simple_state_q":
            if self.simple_proposal_network is None:
                raise RuntimeError("Simple proposal network must be initialized for simple_state_q evaluation")
            eval_collector = SimpleActionCollector(
                env=eval_env,
                replay_buffer=ReplayBuffer(capacity=1, seed=eval_seed_base),
                observation_manager=None,
                config=eval_collector_config,
                proposal_network=self.simple_proposal_network,
                q_network=self.q_network,
            )
        else:
            if self.decoder is None:
                raise RuntimeError("Decoder must be initialized for counterfactual evaluation")
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
                    if self.args.mode == "simple_state_q":
                        next_raw_obs, rewards, dones, infos = eval_env.step(action_selection.selected_action[0])
                    else:
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
        self.log(
            f"eval iter={iteration:05d} | episodes={summary.episodes:03d} | total_reward={summary.total_reward:.3f} | "
            f"mean_reward={summary.mean_reward:.3f} | mean_ep_len={summary.mean_episode_length:.1f} | "
            f"success={summary.success_rate:.3f} | collision={summary.collision_rate:.3f} | timeout={summary.timeout_rate:.3f}"
        )

    def _log_wandb_iteration(self, iteration: int, collect_summary, last_stats, eval_summary: EvaluationSummary | None) -> None:
        if self.wandb_run is None or self.replay_buffer is None or self.start_time is None:
            return

        metrics: dict[str, float | int] = {
            "iteration": int(iteration),
            "replay/size": int(len(self.replay_buffer)),
            "collect/reward": float(collect_summary.total_reward),
            "collect/episodes": int(collect_summary.episodes_completed),
            "collect/transitions": int(collect_summary.transitions_added),
            "time/elapsed_seconds": float(time.time() - self.start_time),
        }
        if last_stats is not None:
            metrics.update(
                {
                    "train/loss": float(last_stats.loss),
                    "train/q_pred_mean": float(last_stats.q_pred_mean),
                    "train/target_mean": float(last_stats.target_mean),
                    "train/next_q_mean": float(last_stats.next_q_mean),
                    "train/selection_entropy_mean": float(last_stats.selection_entropy_mean),
                    "train/reward_mean": float(last_stats.reward_mean),
                    "train/done_fraction": float(last_stats.done_fraction),
                }
            )
        if eval_summary is not None:
            metrics.update(
                {
                    "eval/episodes": int(eval_summary.episodes),
                    "eval/total_reward": float(eval_summary.total_reward),
                    "eval/mean_reward": float(eval_summary.mean_reward),
                    "eval/mean_episode_length": float(eval_summary.mean_episode_length),
                    "eval/success_rate": float(eval_summary.success_rate),
                    "eval/collision_rate": float(eval_summary.collision_rate),
                    "eval/timeout_rate": float(eval_summary.timeout_rate),
                }
            )

        self.wandb_run.log(metrics, step=int(iteration), commit=True)

    def _log_iteration(self, iteration: int, collect_summary, last_stats) -> None:
        if self.replay_buffer is None or self.start_time is None:
            raise RuntimeError("Replay buffer and timer must be initialized before logging")

        elapsed = time.time() - self.start_time
        if last_stats is None:
            self.log(
                f"iter={iteration:05d} | replay={len(self.replay_buffer):06d} | "
                f"collect_reward={collect_summary.total_reward:.3f} | collect_episodes={collect_summary.episodes_completed} | "
                f"updates=0 | elapsed={elapsed:.1f}s"
            )
            return

        self.log(
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
        checkpoint = {
            "iteration": int(iteration),
            "args": vars(self.args),
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "q_model_config": self.q_model_config,
            "trainer_config": asdict(self.trainer_config),
            "collector_config": asdict(self.collector_config),
            "env_config": {
                "max_steps": self.env_config.max_steps,
                "controlled_agent_index": self.env_config.controlled_agent_index,
                "device": self.env_config.device,
            },
            "mode": self.args.mode,
        }
        if self.args.mode == "simple_state_q" and self.simple_proposal_network is not None:
            checkpoint["proposal_network"] = self.simple_proposal_network.state_dict()
        if self.args.mode == "counterfactual_q" and self.args.decoder_checkpoint is not None:
            checkpoint["decoder_checkpoint"] = str(self.args.decoder_checkpoint)
        return checkpoint

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint = self._build_checkpoint(iteration)
        periodic_path = self.output_path.with_name(f"{self.output_path.stem}_iter_{iteration:06d}{self.output_path.suffix}")
        torch.save(checkpoint, periodic_path)
        torch.save(checkpoint, self.output_path)
        self.log(f"Checkpoint saved: {periodic_path}")
        if self.wandb_run is not None:
            self.wandb_run.save(str(periodic_path), base_path=str(self.output_path.parent), policy="now")
            self.wandb_run.save(str(self.output_path), base_path=str(self.output_path.parent), policy="now")

    def _save_final_checkpoint(self) -> None:
        checkpoint = self._build_checkpoint(int(self.args.iterations))
        torch.save(checkpoint, self.output_path)
        self.log(f"Training finished. Final checkpoint saved: {self.output_path}")
        if self.wandb_run is not None:
            self.wandb_run.save(str(self.output_path), base_path=str(self.output_path.parent), policy="now")
