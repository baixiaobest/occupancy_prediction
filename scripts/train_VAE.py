from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.VAE_prediction import VAEPredictionDecoder, VAEPredictionEncoder
from src.VAE_prediction import (
    DEFAULT_DOWNSAMPLE_STRIDES,
    DEFAULT_UPSAMPLE_CHANNELS,
    DEFAULT_UPSAMPLE_STRIDES,
    build_prediction_vae_models,
)
from src.Dataset import build_datasets
from src.loss import (
    bernoulli_entropy_loss,
    kl_divergence,
    kl_target_loss,
    weighted_bernoulli_recon_loss,
    weighted_focal_recon_loss,
)


def get_rollout_length(
    epoch: int,
    start_k: int,
    target_k: int,
    ramp_epochs: int,
) -> int:
    """Linearly increase rollout length from `start_k` to `target_k`."""
    if start_k <= 0 or target_k <= 0:
        raise ValueError("rollout lengths must be > 0")
    if ramp_epochs <= 1:
        return int(target_k)

    progress = float(epoch - 1) / float(ramp_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    value = start_k + progress * (target_k - start_k)
    return max(1, int(round(value)))


def get_teacher_forcing_prob(
    epoch: int,
    start_p: float,
    end_p: float,
    ramp_epochs: int,
) -> float:
    """Linearly decrease teacher forcing probability from `start_p` to `end_p`."""
    if not (0.0 <= start_p <= 1.0 and 0.0 <= end_p <= 1.0):
        raise ValueError("teacher forcing probabilities must be in [0, 1]")
    if ramp_epochs <= 1:
        return float(end_p)

    progress = float(epoch - 1) / float(ramp_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    value = start_p + progress * (end_p - start_p)
    return max(0.0, min(1.0, float(value)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train occupancy prediction VAE")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing rollout .pt files")
    parser.add_argument("--history-len", type=int, default=16, help="Number of past frames as encoder input")
    parser.add_argument("--future-len", type=int, default=8, help="Number of future frames to predict")
    parser.add_argument("--decoder-context-len", type=int, default=8, help="Number of past/current frames used to condition decoder")
    parser.add_argument("--rollout-start-k", type=int, default=1, help="Initial autoregressive rollout length")
    parser.add_argument("--rollout-target-k", type=int, default=8, help="Target autoregressive rollout length")
    parser.add_argument("--num-latent-samples", type=int, default=2, help="Number of sampled z per sample for best-of-N training")
    parser.add_argument(
        "--latent-selection-max-steps",
        type=int,
        default=8,
        help="Max rollout steps used to pick best z (m); full rollout still uses selected z",
    )
    parser.add_argument("--teacher-forcing-start-p", type=float, default=1.0, help="Initial teacher forcing probability")
    parser.add_argument("--teacher-forcing-end-p", type=float, default=0.2, help="Final teacher forcing probability")
    parser.add_argument("--curriculum-epochs", type=int, default=0, help="Epochs to ramp curriculum (0 uses total epochs)")
    parser.add_argument("--window-stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for origin-level split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--recon-loss-type",
        type=str,
        default="focal",
        choices=("bce", "focal"),
        help="Reconstruction loss type",
    )
    parser.add_argument(
        "--occupied-weight",
        type=float,
        default=5.0,
        help="Absolute occupied-cell weight for BCE/focal losses (>=1)",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (0 disables focal modulation)")
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help="Weight for per-cell Bernoulli entropy regularization over predicted frames",
    )
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--target-kl", type=float, default=1.0, help="Target KL value used in quadratic KL objective")
    parser.add_argument("--latent-channel", type=int, default=128)
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 128, 128, 128],
        help="Encoder channel widths. Must have len(downsample_strides)+1 entries.",
    )
    parser.add_argument("--decoder-base-channels", type=int, default=8)
    parser.add_argument(
        "--decoder-downsample-channels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 128, 128, 128],
        help="Decoder context downsample channel widths. Must have len(DEFAULT_DOWNSAMPLE_STRIDES)+1 entries.",
    )
    parser.add_argument(
        "--decoder-context-latent-channel",
        type=int,
        default=32,
        help="Context branch latent channels in decoder (defaults to latent-channel)",
    )
    parser.add_argument("--static-stem-channels", type=int, default=8)
    parser.add_argument(
        "--velocity-mlp-dim",
        type=int,
        default=16,
        help="Velocity embedding dimension C after MLP",
    )
    parser.add_argument(
        "--encoder-velocity-condition-channels",
        type=int,
        default=4,
        help="Velocity conditioning channel count C1 fused in encoder",
    )
    parser.add_argument(
        "--decoder-velocity-condition-channels",
        type=int,
        default=2,
        help="Velocity conditioning channel count C2 fused in decoder context branch",
    )
    parser.add_argument(
        "--decoder-position-mlp-dim",
        type=int,
        default=16,
        help="Position-offset embedding dimension after MLP",
    )
    parser.add_argument(
        "--decoder-position-condition-channels",
        type=int,
        default=2,
        help="Position-offset conditioning channel count C3 fused in decoder context branch",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--lazy-data-load",
        action="store_true",
        help="Enable lazy on-the-fly occupancy window slicing during dataset loading",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vae_prediction.pt"))
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save an epoch checkpoint every N epochs",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable Weights & Biases logging",
    )
    parser.add_argument("--wandb-project", type=str, default="occupancy-prediction")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _format_hhmmss(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_message(message: str, wandb_run: object | None = None) -> None:
    """Log to terminal reliably and mirror to W&B when enabled."""
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    stream.write(f"{message}\n")
    stream.flush()

    if wandb_run is not None and wandb is not None:
        wandb.termlog(message)


class VAETrainer:
    """Owns training state and runs the end-to-end train/validate loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._validate_args()

        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.output.parent.mkdir(parents=True, exist_ok=True)

        self.wandb_run: object | None = self._init_wandb()
        self.curriculum_epochs = (
            self.args.epochs if self.args.curriculum_epochs <= 0 else self.args.curriculum_epochs
        )

        self.downsample_strides = list(DEFAULT_DOWNSAMPLE_STRIDES)
        self.upsample_strides = list(DEFAULT_UPSAMPLE_STRIDES)
        self.upsample_channels = list(DEFAULT_UPSAMPLE_CHANNELS)

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.input_shape: tuple[int, int, int, int]
        self.output_shape: tuple[int, int, int, int]
        self.decoder_context_frames: int

        self.encoder: VAEPredictionEncoder
        self.decoder: VAEPredictionDecoder
        self.optimizer: torch.optim.Optimizer
        self.decoder_context_latent_channel: int

        self._build_data()
        self._build_models_and_optimizer()

    def _validate_args(self) -> None:
        if self.args.decoder_context_len > self.args.history_len:
            raise ValueError("decoder_context_len must be <= history_len")
        if self.args.save_interval <= 0:
            raise ValueError("save_interval must be > 0")
        if self.args.rollout_start_k <= 0 or self.args.rollout_target_k <= 0:
            raise ValueError("rollout_start_k and rollout_target_k must be > 0")
        if self.args.num_latent_samples <= 0:
            raise ValueError("num_latent_samples must be > 0")
        if self.args.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be >= 0")
        if self.args.latent_selection_max_steps <= 0:
            raise ValueError("latent_selection_max_steps must be > 0")
        if not (
            0.0 <= self.args.teacher_forcing_start_p <= 1.0
            and 0.0 <= self.args.teacher_forcing_end_p <= 1.0
        ):
            raise ValueError("teacher forcing probabilities must be in [0, 1]")

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

    def _build_data(self) -> None:
        self.log("Building datasets...")
        train_dataset, val_dataset, stats = build_datasets(
            data_dir=self.args.data_dir,
            val_ratio=self.args.val_ratio,
            history_len=self.args.history_len,
            future_len=self.args.future_len,
            decoder_context_len=self.args.decoder_context_len,
            window_stride=self.args.window_stride,
            seed=self.args.seed,
            lazy=self.args.lazy_data_load,
        )
        self.log("Datasets built.")

        if len(train_dataset) == 0:
            raise ValueError("No training samples were created. Check data and window parameters.")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        (
            sample_x_encoder_dynamic,
            sample_x_decoder_dynamic,
            _sample_x_static,
            _sample_current_velocity,
            _sample_future_velocities,
            _sample_future_position_offsets,
            sample_y,
        ) = train_dataset[0]
        _, enc_t, h, w = sample_x_encoder_dynamic.shape
        _, dec_ctx_t, _, _ = sample_x_decoder_dynamic.shape
        self.input_shape = (1, enc_t, h, w)
        self.output_shape = (1, sample_y.shape[1], h, w)
        self.decoder_context_frames = dec_ctx_t

        self.log(
            "Dataset summary: "
            f"files={stats.num_scene_files}, "
            f"train_agent_sequences={stats.num_train_agent_sequences}, "
            f"val_agent_sequences={stats.num_val_agent_sequences}, "
            f"train_anchors={stats.num_train_anchors}, val_anchors={stats.num_val_anchors}, "
            f"train_samples={stats.num_train_samples}, val_samples={stats.num_val_samples}"
        )

    def _build_models_and_optimizer(self) -> None:
        if len(self.args.channels) != len(self.downsample_strides) + 1:
            raise ValueError("channels must have len(DEFAULT_DOWNSAMPLE_STRIDES)+1 entries")
        if (
            self.args.decoder_downsample_channels is not None
            and len(self.args.decoder_downsample_channels) != len(self.downsample_strides) + 1
        ):
            raise ValueError(
                "decoder_downsample_channels must have len(DEFAULT_DOWNSAMPLE_STRIDES)+1 entries"
            )

        self.decoder_context_latent_channel = (
            self.args.decoder_context_latent_channel
            if self.args.decoder_context_latent_channel is not None
            else self.args.latent_channel
        )

        self.encoder, self.decoder = build_prediction_vae_models(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            latent_channel=self.args.latent_channel,
            channels=self.args.channels,
            decoder_downsample_channels=self.args.decoder_downsample_channels,
            decoder_context_latent_channel=self.decoder_context_latent_channel,
            static_stem_channels=self.args.static_stem_channels,
            velocity_mlp_dim=self.args.velocity_mlp_dim,
            encoder_velocity_condition_channels=self.args.encoder_velocity_condition_channels,
            decoder_velocity_condition_channels=self.args.decoder_velocity_condition_channels,
            decoder_position_mlp_dim=self.args.decoder_position_mlp_dim,
            decoder_position_condition_channels=self.args.decoder_position_condition_channels,
            decoder_context_frames=self.decoder_context_frames,
            downsample_strides=self.downsample_strides,
            decoder_context_downsample_strides=self.downsample_strides,
            upsample_strides=self.upsample_strides,
            upsample_channels=self.upsample_channels,
            device=self.device,
        )

        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def _build_checkpoint(
        self,
        *,
        epoch: int,
        best_epoch: int,
        best_val_loss: float,
        train_loss: float,
        val_loss: float,
        train_kl: float,
        val_kl: float,
    ) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "args": vars(self.args),
            "model_config": {
                "history_len": self.args.history_len,
                "future_len": self.args.future_len,
                "decoder_context_len": self.args.decoder_context_len,
                "rollout_start_k": self.args.rollout_start_k,
                "rollout_target_k": self.args.rollout_target_k,
                "num_latent_samples": self.args.num_latent_samples,
                "latent_selection_max_steps": self.args.latent_selection_max_steps,
                "teacher_forcing_start_p": self.args.teacher_forcing_start_p,
                "teacher_forcing_end_p": self.args.teacher_forcing_end_p,
                "target_kl": self.args.target_kl,
                "curriculum_epochs": self.curriculum_epochs,
                "latent_channel": self.args.latent_channel,
                "channels": list(self.args.channels),
                "decoder_base_channels": self.args.decoder_base_channels,
                "decoder_downsample_channels": (
                    list(self.args.decoder_downsample_channels)
                    if self.args.decoder_downsample_channels is not None
                    else None
                ),
                "decoder_context_latent_channel": self.decoder_context_latent_channel,
                "static_stem_channels": self.args.static_stem_channels,
                "velocity_mlp_dim": self.args.velocity_mlp_dim,
                "encoder_velocity_condition_channels": self.args.encoder_velocity_condition_channels,
                "decoder_velocity_condition_channels": self.args.decoder_velocity_condition_channels,
                "decoder_position_mlp_dim": self.args.decoder_position_mlp_dim,
                "decoder_position_condition_channels": self.args.decoder_position_condition_channels,
                "downsample_strides": self.downsample_strides,
                "upsample_strides": self.upsample_strides,
                "upsample_channels": self.upsample_channels,
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
            },
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_kl": train_kl,
            "val_kl": val_kl,
        }

    def _rollout_step(
        self,
        *,
        z_step: torch.Tensor,
        context_step: torch.Tensor,
        static_step: torch.Tensor,
        step_velocity: torch.Tensor,
        step_position_offset: torch.Tensor,
        target_step: torch.Tensor,
        is_train: bool,
        teacher_forcing_prob: float,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one autoregressive decoder step and return per-sample losses + next context."""
        logits_full = self.decoder(
            z_step,
            context_step,
            static_step,
            step_velocity,
            step_position_offset,
        )
        logits_step = logits_full[:, :, :1]

        if self.args.recon_loss_type == "focal":
            step_recon_cells = weighted_focal_recon_loss(
                logits_step,
                target_step,
                occupied_weight=self.args.occupied_weight,
                focal_gamma=self.args.focal_gamma,
                reduction="none",
            )
        elif self.args.recon_loss_type == "bce":
            step_recon_cells = weighted_bernoulli_recon_loss(
                logits_step,
                target_step,
                occupied_weight=self.args.occupied_weight,
                reduction="none",
            )
        else:
            raise ValueError(f"Unsupported recon_loss_type: {self.args.recon_loss_type}")

        step_recon = step_recon_cells.reshape(batch_size, -1).mean(dim=1)
        step_entropy_cells = bernoulli_entropy_loss(logits_step, reduction="none")
        step_entropy = step_entropy_cells.reshape(batch_size, -1).mean(dim=1)

        pred_step = torch.sigmoid(logits_step.detach())
        if is_train and teacher_forcing_prob < 1.0:
            # Scheduled sampling is applied independently for each sample.
            use_gt = (
                torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < teacher_forcing_prob
            ).to(target_step.dtype)
            feedback_step = use_gt * target_step + (1.0 - use_gt) * pred_step
        else:
            feedback_step = target_step if teacher_forcing_prob >= 1.0 else pred_step

        # Keep decoder context in the same anchor-centered frame across rollout.
        next_context = torch.cat([context_step[:, :, 1:], feedback_step], dim=2)
        return step_recon, step_entropy, next_context

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        *,
        rollout_len: int,
        teacher_forcing_prob: float,
    ) -> tuple[float, float, float, float, float]:
        """Run one epoch over `loader` for train or eval based on optimizer."""
        is_train = optimizer is not None
        self.encoder.train(is_train)
        self.decoder.train(is_train)

        total_loss = 0.0
        total_recon = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_kl_objective = 0.0
        total_batches = 0

        for (
            x_encoder_dynamic,
            x_decoder_dynamic,
            x_static,
            current_velocity,
            future_velocities,
            future_position_offsets,
            y,
        ) in loader:
            x_encoder_dynamic = x_encoder_dynamic.to(self.device)
            x_decoder_dynamic = x_decoder_dynamic.to(self.device)
            x_static = x_static.to(self.device)
            current_velocity = current_velocity.to(self.device)
            future_velocities = future_velocities.to(self.device)
            future_position_offsets = future_position_offsets.to(self.device)
            y = y.to(self.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            mu, sigma = self.encoder(x_encoder_dynamic, x_static, current_velocity)

            horizon = int(y.shape[2])
            effective_k = max(1, min(int(rollout_len), horizon))
            selection_steps = max(1, min(effective_k, int(self.args.latent_selection_max_steps)))
            batch_size = y.shape[0]

            selection_step_recons = torch.empty(
                (self.args.num_latent_samples, batch_size, selection_steps),
                device=self.device,
                dtype=y.dtype,
            )
            selection_step_entropies = torch.empty(
                (self.args.num_latent_samples, batch_size, selection_steps),
                device=self.device,
                dtype=y.dtype,
            )
            z_candidates = torch.empty(
                (self.args.num_latent_samples,) + tuple(mu.shape),
                device=self.device,
                dtype=mu.dtype,
            )
            context_candidates = torch.empty(
                (self.args.num_latent_samples,) + tuple(x_decoder_dynamic.shape),
                device=self.device,
                dtype=x_decoder_dynamic.dtype,
            )

            for z_idx in range(self.args.num_latent_samples):
                z = self.encoder.sample(mu, sigma)
                z_candidates[z_idx] = z
                context = x_decoder_dynamic.clone()

                for step in range(selection_steps):
                    step_velocity = future_velocities[:, step, :]
                    step_position_offset = future_position_offsets[:, step, :]
                    static_step = x_static[:, :, self.args.history_len + step]
                    target_step = y[:, :, step : step + 1]
                    step_recon, step_entropy, context = self._rollout_step(
                        z_step=z,
                        context_step=context,
                        static_step=static_step,
                        step_velocity=step_velocity,
                        step_position_offset=step_position_offset,
                        target_step=target_step,
                        is_train=is_train,
                        teacher_forcing_prob=teacher_forcing_prob,
                        batch_size=batch_size,
                    )
                    selection_step_recons[z_idx, :, step] = step_recon
                    selection_step_entropies[z_idx, :, step] = step_entropy

                context_candidates[z_idx] = context

            z_selection_losses = selection_step_recons.mean(dim=2)
            best_z_indices = z_selection_losses.argmin(dim=0)
            batch_indices = torch.arange(batch_size, device=self.device)

            selected_step_recons = selection_step_recons[best_z_indices, batch_indices, :]
            selected_step_entropies = selection_step_entropies[best_z_indices, batch_indices, :]

            if effective_k > selection_steps:
                rollout_step_recons = torch.empty(
                    (batch_size, effective_k),
                    device=self.device,
                    dtype=y.dtype,
                )
                rollout_step_entropies = torch.empty(
                    (batch_size, effective_k),
                    device=self.device,
                    dtype=y.dtype,
                )
                rollout_step_recons[:, :selection_steps] = selected_step_recons
                rollout_step_entropies[:, :selection_steps] = selected_step_entropies

                best_z = z_candidates[best_z_indices, batch_indices, ...]
                context = context_candidates[best_z_indices, batch_indices, ...]

                for step in range(selection_steps, effective_k):
                    step_velocity = future_velocities[:, step, :]
                    step_position_offset = future_position_offsets[:, step, :]
                    static_step = x_static[:, :, self.args.history_len + step]
                    target_step = y[:, :, step : step + 1]
                    step_recon, step_entropy, context = self._rollout_step(
                        z_step=best_z,
                        context_step=context,
                        static_step=static_step,
                        step_velocity=step_velocity,
                        step_position_offset=step_position_offset,
                        target_step=target_step,
                        is_train=is_train,
                        teacher_forcing_prob=teacher_forcing_prob,
                        batch_size=batch_size,
                    )
                    rollout_step_recons[:, step] = step_recon
                    rollout_step_entropies[:, step] = step_entropy
            else:
                rollout_step_recons = selected_step_recons
                rollout_step_entropies = selected_step_entropies

            recon = rollout_step_recons.mean()
            entropy = rollout_step_entropies.mean()
            kl = kl_divergence(mu, sigma)
            kl_objective = kl_target_loss(kl, target_kl=self.args.target_kl)
            loss = recon + self.args.entropy_weight * entropy + self.args.kl_weight * kl_objective

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_entropy += float(entropy.item())
            total_kl += float(kl.item())
            total_kl_objective += float(kl_objective.item())
            total_batches += 1

        if total_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            total_loss / total_batches,
            total_recon / total_batches,
            total_entropy / total_batches,
            total_kl / total_batches,
            total_kl_objective / total_batches,
        )

    def train(self) -> None:
        total_epoch_time = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        self.log(f"total epochs: {self.args.epochs}")

        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.perf_counter()

            current_rollout_k = get_rollout_length(
                epoch=epoch,
                start_k=self.args.rollout_start_k,
                target_k=self.args.rollout_target_k,
                ramp_epochs=self.curriculum_epochs,
            )
            current_rollout_k = min(current_rollout_k, self.args.future_len)
            current_teacher_forcing_p = get_teacher_forcing_prob(
                epoch=epoch,
                start_p=self.args.teacher_forcing_start_p,
                end_p=self.args.teacher_forcing_end_p,
                ramp_epochs=self.curriculum_epochs,
            )

            train_loss, train_recon, train_entropy, train_kl, train_kl_objective = self.run_epoch(
                self.train_loader,
                self.optimizer,
                rollout_len=current_rollout_k,
                teacher_forcing_prob=current_teacher_forcing_p,
            )

            with torch.no_grad():
                val_loss, val_recon, val_entropy, val_kl, val_kl_objective = self.run_epoch(
                    self.val_loader,
                    optimizer=None,
                    rollout_len=current_rollout_k,
                    teacher_forcing_prob=current_teacher_forcing_p,
                )

            epoch_duration = time.perf_counter() - epoch_start
            total_epoch_time += epoch_duration
            avg_epoch_duration = total_epoch_time / epoch
            eta_seconds = avg_epoch_duration * (self.args.epochs - epoch)

            self.log(
                f"Epoch {epoch:03d} | "
                f"K={current_rollout_k}, tf_p={current_teacher_forcing_p:.3f}, "
                f"n_z={self.args.num_latent_samples}, m={self.args.latent_selection_max_steps} | "
                f"train: loss={train_loss:.6f}, recon={train_recon:.6f}, entropy={train_entropy:.6f}, kl={train_kl:.6f}, kl_obj={train_kl_objective:.6f} | "
                f"val: loss={val_loss:.6f}, recon={val_recon:.6f}, entropy={val_entropy:.6f}, kl={val_kl:.6f}, kl_obj={val_kl_objective:.6f} | "
                f"epoch_time={_format_hhmmss(epoch_duration)}, eta={_format_hhmmss(eta_seconds)}"
            )

            if epoch % self.args.save_interval == 0:
                periodic_ckpt = (
                    self.args.output.parent
                    / f"{self.args.output.stem}_periodic_epoch_{epoch:03d}{self.args.output.suffix}"
                )
                periodic_checkpoint = self._build_checkpoint(
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_kl=train_kl,
                    val_kl=val_kl,
                )
                torch.save(periodic_checkpoint, periodic_ckpt)
                self.log(f"Periodic checkpoint saved: {periodic_ckpt}")

                if self.wandb_run is not None:
                    self.wandb_run.save(
                        str(periodic_ckpt),
                        base_path=str(self.args.output.parent),
                        policy="now",
                    )

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/recon": train_recon,
                        "train/entropy": train_entropy,
                        "train/kl": train_kl,
                        "train/kl_objective": train_kl_objective,
                        "val/loss": val_loss,
                        "val/recon": val_recon,
                        "val/entropy": val_entropy,
                        "val/kl": val_kl,
                        "val/kl_objective": val_kl_objective,
                        "loss/entropy_weight": self.args.entropy_weight,
                        "kl/target": self.args.target_kl,
                        "curriculum/rollout_k": current_rollout_k,
                        "curriculum/teacher_forcing_p": current_teacher_forcing_p,
                        "best/val_loss": min(best_val_loss, val_loss),
                    },
                    step=epoch,
                    commit=True,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                epoch_ckpt = self.args.output.parent / f"{self.args.output.stem}_epoch_{epoch:03d}{self.args.output.suffix}"
                checkpoint = self._build_checkpoint(
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_kl=train_kl,
                    val_kl=val_kl,
                )
                torch.save(checkpoint, epoch_ckpt)
                torch.save(checkpoint, self.args.output)
                self.log(
                    f"Validation improved to {best_val_loss:.6f}; "
                    f"saved checkpoint: {epoch_ckpt}"
                )

                if self.wandb_run is not None:
                    self.wandb_run.save(str(epoch_ckpt), base_path=str(self.args.output.parent), policy="now")
                    self.wandb_run.save(str(self.args.output), base_path=str(self.args.output.parent), policy="now")

                    artifact = wandb.Artifact(
                        name=f"vae-checkpoint-{epoch}",
                        type="model",
                        metadata={"epoch": epoch, "best_val_loss": best_val_loss},
                    )
                    artifact.add_file(str(epoch_ckpt), name=epoch_ckpt.name)
                    artifact.add_file(str(self.args.output), name=self.args.output.name)
                    self.wandb_run.log_artifact(
                        artifact,
                        aliases=["best", "latest", f"epoch-{epoch:03d}"],
                    )
            else:
                self.log(
                    f"No validation improvement (best={best_val_loss:.6f} at epoch {best_epoch:03d}); "
                    "checkpoint not updated."
                )

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def main() -> None:
    trainer = VAETrainer(parse_args())
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
