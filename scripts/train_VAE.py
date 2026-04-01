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


def run_epoch(
    encoder: VAEPredictionEncoder,
    decoder: VAEPredictionDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    recon_loss_type: str,
    occupied_weight: float,
    focal_gamma: float,
    entropy_weight: float,
    kl_weight: float,
    target_kl: float,
    rollout_len: int,
    teacher_forcing_prob: float,
    num_latent_samples: int,
    latent_selection_max_steps: int,
    z_separation_weight: float,
) -> tuple[float, float, float, float, float, float]:
    """Run one epoch (training or evaluation) over `loader`.

    If `optimizer` is provided the model parameters are updated (training
    mode). Otherwise the models are run in evaluation mode and gradients are
    not applied.

    Args:
        encoder: VAE encoder instance.
        decoder: VAE decoder instance.
        loader: DataLoader yielding
            `(x_encoder_dynamic, x_decoder_dynamic, x_static, current_velocity, y)`.
        optimizer: Optimizer to use for training, or `None` for evaluation.
        device: Device to run tensors on.
        recon_loss_type: Reconstruction loss type: `bce` or `focal`.
        occupied_weight: Absolute occupied-cell weight for both BCE and focal losses.
        focal_gamma: Focal exponent used to focus on hard examples.
        entropy_weight: Weight for predictive entropy regularization.
        kl_weight: Weight to scale the KL objective when computing total loss.
        target_kl: Target KL value used in quadratic penalty `(KL - target_kl)^2`.
        rollout_len: Number of autoregressive steps used for reconstruction loss.
        teacher_forcing_prob: Probability of feeding GT at each rollout step.
        num_latent_samples: Number of latent samples (`z`) to evaluate per sample.
        latent_selection_max_steps: Max rollout steps used to choose best `z`.
        z_separation_weight: Weight for latent/prediction separation regularizer.

    Returns:
        A tuple `(avg_loss, avg_recon, avg_entropy, avg_kl, avg_kl_objective, avg_z_separation)` averaged over batches.
    """
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)

    total_loss = 0.0
    total_recon = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_kl_objective = 0.0
    total_z_separation = 0.0
    total_batches = 0

    for x_encoder_dynamic, x_decoder_dynamic, x_static, current_velocity, y in loader:
        x_encoder_dynamic = x_encoder_dynamic.to(device)
        x_decoder_dynamic = x_decoder_dynamic.to(device)
        x_static = x_static.to(device)
        current_velocity = current_velocity.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        mu, sigma = encoder(x_encoder_dynamic, x_static, current_velocity)

        horizon = int(y.shape[2])
        effective_k = max(1, min(int(rollout_len), horizon))
        selection_steps = max(1, min(effective_k, int(latent_selection_max_steps)))
        batch_size = y.shape[0]

        def _rollout_step(
            z_step: torch.Tensor,
            context_step: torch.Tensor,
            step: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_full = decoder(z_step, context_step, x_static, current_velocity)
            logits_step = logits_full[:, :, :1]
            target_step = y[:, :, step : step + 1]

            if recon_loss_type == "focal":
                step_recon_cells = weighted_focal_recon_loss(
                    logits_step,
                    target_step,
                    occupied_weight=occupied_weight,
                    focal_gamma=focal_gamma,
                    reduction="none",
                )
            elif recon_loss_type == "bce":
                step_recon_cells = weighted_bernoulli_recon_loss(
                    logits_step,
                    target_step,
                    occupied_weight=occupied_weight,
                    reduction="none",
                )
            else:
                raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

            step_recon = step_recon_cells.reshape(batch_size, -1).mean(dim=1)
            step_entropy_cells = bernoulli_entropy_loss(logits_step, reduction="none")
            step_entropy = step_entropy_cells.reshape(batch_size, -1).mean(dim=1)

            pred_step = torch.sigmoid(logits_step.detach())
            if is_train and teacher_forcing_prob < 1.0:
                use_gt = (
                    torch.rand((batch_size, 1, 1, 1, 1), device=device) < teacher_forcing_prob
                ).to(target_step.dtype)
                feedback_step = use_gt * target_step + (1.0 - use_gt) * pred_step
            else:
                feedback_step = target_step if teacher_forcing_prob >= 1.0 else pred_step

            next_context = torch.cat([context_step[:, :, 1:], feedback_step], dim=2)
            return step_recon, step_entropy, next_context, torch.sigmoid(logits_step)

        selection_step_recons = torch.empty(
            (num_latent_samples, batch_size, selection_steps),
            device=device,
            dtype=y.dtype,
        )
        selection_step_entropies = torch.empty(
            (num_latent_samples, batch_size, selection_steps),
            device=device,
            dtype=y.dtype,
        )
        z_candidates = torch.empty(
            (num_latent_samples,) + tuple(mu.shape),
            device=device,
            dtype=mu.dtype,
        )
        context_candidates = torch.empty(
            (num_latent_samples,) + tuple(x_decoder_dynamic.shape),
            device=device,
            dtype=x_decoder_dynamic.dtype,
        )
        selection_step_probs = torch.empty(
            (num_latent_samples, batch_size, selection_steps, y.shape[1], y.shape[3], y.shape[4]),
            device=device,
            dtype=y.dtype,
        )

        for z_idx in range(num_latent_samples):
            z = encoder.sample(mu, sigma)
            z_candidates[z_idx] = z
            context = x_decoder_dynamic.clone()

            for step in range(selection_steps):
                step_recon, step_entropy, context, step_prob = _rollout_step(z, context, step)
                selection_step_recons[z_idx, :, step] = step_recon
                selection_step_entropies[z_idx, :, step] = step_entropy
                selection_step_probs[z_idx, :, step] = step_prob.squeeze(2)

            context_candidates[z_idx] = context

        if z_separation_weight > 0.0 and num_latent_samples > 1:
            z_flat = z_candidates.reshape(num_latent_samples, batch_size, -1)
            probs_flat = selection_step_probs.reshape(num_latent_samples, batch_size, -1)

            pred_diff_sq = (probs_flat.unsqueeze(1) - probs_flat.unsqueeze(0)).square().mean(dim=-1)
            z_dist = (z_flat.unsqueeze(1) - z_flat.unsqueeze(0)).abs().mean(dim=-1)

            pair_i, pair_j = torch.triu_indices(num_latent_samples, num_latent_samples, offset=1, device=device)
            pairwise_weighted_sep = z_dist[pair_i, pair_j] * pred_diff_sq[pair_i, pair_j]
            z_separation = pairwise_weighted_sep.mean()
            z_separation_loss = -z_separation_weight * z_separation
        else:
            z_separation = torch.zeros((), device=device, dtype=y.dtype)
            z_separation_loss = torch.zeros((), device=device, dtype=y.dtype)

        z_selection_losses = selection_step_recons.mean(dim=2)
        best_z_indices = z_selection_losses.argmin(dim=0)
        batch_indices = torch.arange(batch_size, device=device)

        selected_step_recons = selection_step_recons[best_z_indices, batch_indices, :]
        selected_step_entropies = selection_step_entropies[best_z_indices, batch_indices, :]

        if effective_k > selection_steps:
            rollout_step_recons = torch.empty(
                (batch_size, effective_k),
                device=device,
                dtype=y.dtype,
            )
            rollout_step_entropies = torch.empty(
                (batch_size, effective_k),
                device=device,
                dtype=y.dtype,
            )
            rollout_step_recons[:, :selection_steps] = selected_step_recons
            rollout_step_entropies[:, :selection_steps] = selected_step_entropies

            best_z = z_candidates[best_z_indices, batch_indices, ...]
            context = context_candidates[best_z_indices, batch_indices, ...]

            for step in range(selection_steps, effective_k):
                step_recon, step_entropy, context = _rollout_step(best_z, context, step)
                rollout_step_recons[:, step] = step_recon
                rollout_step_entropies[:, step] = step_entropy
        else:
            rollout_step_recons = selected_step_recons
            rollout_step_entropies = selected_step_entropies

        del selection_step_probs

        recon = rollout_step_recons.mean()
        entropy = rollout_step_entropies.mean()
        kl = kl_divergence(mu, sigma)
        kl_objective = kl_target_loss(kl, target_kl=target_kl)
        loss = recon + entropy_weight * entropy + kl_weight * kl_objective + z_separation_loss

        del selection_step_recons, selection_step_entropies, z_candidates, context_candidates

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_entropy += float(entropy.item())
        total_kl += float(kl.item())
        total_kl_objective += float(kl_objective.item())
        total_z_separation += float(z_separation.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / total_batches,
        total_recon / total_batches,
        total_entropy / total_batches,
        total_kl / total_batches,
        total_kl_objective / total_batches,
        total_z_separation / total_batches,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed argparse namespace with training and model hyperparameters.
    """
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
    parser.add_argument(
        "--z-separation-weight",
        type=float,
        default=0.0,
        help="Weight for encouraging different sampled z to produce different predictions",
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
        default=4,
        help="Velocity conditioning channel count C2 fused in decoder context branch",
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
    """Log to terminal reliably and mirror to W&B when enabled.

    Using ``sys.__stdout__`` bypasses wrappers that can swallow output in
    hosted environments.
    """
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    stream.write(f"{message}\n")
    stream.flush()

    if wandb_run is not None and wandb is not None:
        wandb.termlog(message)


def build_checkpoint(
    encoder: VAEPredictionEncoder,
    decoder: VAEPredictionDecoder,
    args: argparse.Namespace,
    curriculum_epochs: int,
    decoder_context_latent_channel: int,
    downsample_strides: Sequence[tuple[int, int]],
    upsample_strides: Sequence[tuple[int, int]],
    upsample_channels: Sequence[int],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    train_loss: float,
    val_loss: float,
    train_kl: float,
    val_kl: float,
) -> dict:
    return {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "args": vars(args),
        "model_config": {
            "history_len": args.history_len,
            "future_len": args.future_len,
            "decoder_context_len": args.decoder_context_len,
            "rollout_start_k": args.rollout_start_k,
            "rollout_target_k": args.rollout_target_k,
            "num_latent_samples": args.num_latent_samples,
            "latent_selection_max_steps": args.latent_selection_max_steps,
            "teacher_forcing_start_p": args.teacher_forcing_start_p,
            "teacher_forcing_end_p": args.teacher_forcing_end_p,
            "target_kl": args.target_kl,
            "z_separation_weight": args.z_separation_weight,
            "curriculum_epochs": curriculum_epochs,
            "latent_channel": args.latent_channel,
            "channels": list(args.channels),
            "decoder_base_channels": args.decoder_base_channels,
            "decoder_downsample_channels": (
                list(args.decoder_downsample_channels)
                if args.decoder_downsample_channels is not None
                else None
            ),
            "decoder_context_latent_channel": decoder_context_latent_channel,
            "static_stem_channels": args.static_stem_channels,
            "velocity_mlp_dim": args.velocity_mlp_dim,
            "encoder_velocity_condition_channels": args.encoder_velocity_condition_channels,
            "decoder_velocity_condition_channels": args.decoder_velocity_condition_channels,
            "downsample_strides": downsample_strides,
            "upsample_strides": upsample_strides,
            "upsample_channels": upsample_channels,
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_kl": train_kl,
        "val_kl": val_kl,
    }


def main() -> None:
    """Entry point for training the VAE model.

    Performs dataset discovery, constructs the VAE encoder/decoder, runs the
    training loop for the specified number of epochs, and saves the best
    checkpoint to `args.output`.
    """
    args = parse_args()

    if args.decoder_context_len > args.history_len:
        raise ValueError("decoder_context_len must be <= history_len")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be > 0")
    if args.rollout_start_k <= 0 or args.rollout_target_k <= 0:
        raise ValueError("rollout_start_k and rollout_target_k must be > 0")
    if args.num_latent_samples <= 0:
        raise ValueError("num_latent_samples must be > 0")
    if args.entropy_weight < 0.0:
        raise ValueError("entropy_weight must be >= 0")
    if args.z_separation_weight < 0.0:
        raise ValueError("z_separation_weight must be >= 0")
    if args.latent_selection_max_steps <= 0:
        raise ValueError("latent_selection_max_steps must be > 0")
    if not (0.0 <= args.teacher_forcing_start_p <= 1.0 and 0.0 <= args.teacher_forcing_end_p <= 1.0):
        raise ValueError("teacher forcing probabilities must be in [0, 1]")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            config=vars(args),
        )

    _log_message("Building datasets...", wandb_run=wandb_run)
    train_dataset, val_dataset, stats = build_datasets(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        history_len=args.history_len,
        future_len=args.future_len,
        decoder_context_len=args.decoder_context_len,
        window_stride=args.window_stride,
        seed=args.seed,
        lazy=args.lazy_data_load,
    )
    _log_message(f"Datasets built.", wandb_run=wandb_run)

    if len(train_dataset) == 0:
        raise ValueError("No training samples were created. Check data and window parameters.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sample_x_encoder_dynamic, sample_x_decoder_dynamic, sample_x_static, _sample_current_velocity, sample_y = train_dataset[0]
    _, enc_t, h, w = sample_x_encoder_dynamic.shape
    _, dec_ctx_t, _, _ = sample_x_decoder_dynamic.shape
    input_shape = (1, enc_t, h, w)
    output_shape = (1, sample_y.shape[1], h, w)

    downsample_strides = list(DEFAULT_DOWNSAMPLE_STRIDES)
    upsample_strides = list(DEFAULT_UPSAMPLE_STRIDES)
    upsample_channels = list(DEFAULT_UPSAMPLE_CHANNELS)
    if len(args.channels) != len(downsample_strides) + 1:
        raise ValueError("channels must have len(DEFAULT_DOWNSAMPLE_STRIDES)+1 entries")
    if args.decoder_downsample_channels is not None and len(args.decoder_downsample_channels) != len(downsample_strides) + 1:
        raise ValueError("decoder_downsample_channels must have len(DEFAULT_DOWNSAMPLE_STRIDES)+1 entries")

    decoder_context_latent_channel = (
        args.decoder_context_latent_channel
        if args.decoder_context_latent_channel is not None
        else args.latent_channel
    )

    encoder, decoder = build_prediction_vae_models(
        input_shape=input_shape,
        output_shape=output_shape,
        latent_channel=args.latent_channel,
        channels=args.channels,
        decoder_downsample_channels=args.decoder_downsample_channels,
        decoder_context_latent_channel=decoder_context_latent_channel,
        static_stem_channels=args.static_stem_channels,
        velocity_mlp_dim=args.velocity_mlp_dim,
        encoder_velocity_condition_channels=args.encoder_velocity_condition_channels,
        decoder_velocity_condition_channels=args.decoder_velocity_condition_channels,
        decoder_context_frames=dec_ctx_t,
        downsample_strides=downsample_strides,
        decoder_context_downsample_strides=downsample_strides,
        upsample_strides=upsample_strides,
        upsample_channels=upsample_channels,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    _log_message(
        "Dataset summary: "
        f"files={stats.num_scene_files}, "
        f"train_agent_sequences={stats.num_train_agent_sequences}, "
        f"val_agent_sequences={stats.num_val_agent_sequences}, "
        f"train_anchors={stats.num_train_anchors}, val_anchors={stats.num_val_anchors}, "
        f"train_samples={stats.num_train_samples}, val_samples={stats.num_val_samples}",
    )

    try:
        total_epoch_time = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        curriculum_epochs = args.epochs if args.curriculum_epochs <= 0 else args.curriculum_epochs
        print(f"total epochs: {args.epochs}")
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()

            current_rollout_k = get_rollout_length(
                epoch=epoch,
                start_k=args.rollout_start_k,
                target_k=args.rollout_target_k,
                ramp_epochs=curriculum_epochs,
            )
            current_rollout_k = min(current_rollout_k, args.future_len)
            current_teacher_forcing_p = get_teacher_forcing_prob(
                epoch=epoch,
                start_p=args.teacher_forcing_start_p,
                end_p=args.teacher_forcing_end_p,
                ramp_epochs=curriculum_epochs,
            )

            train_loss, train_recon, train_entropy, train_kl, train_kl_objective, train_z_separation = run_epoch(
                encoder,
                decoder,
                train_loader,
                optimizer,
                device,
                recon_loss_type=args.recon_loss_type,
                occupied_weight=args.occupied_weight,
                focal_gamma=args.focal_gamma,
                entropy_weight=args.entropy_weight,
                kl_weight=args.kl_weight,
                target_kl=args.target_kl,
                rollout_len=current_rollout_k,
                teacher_forcing_prob=current_teacher_forcing_p,
                num_latent_samples=args.num_latent_samples,
                latent_selection_max_steps=args.latent_selection_max_steps,
                z_separation_weight=args.z_separation_weight,
            )

            with torch.no_grad():
                val_loss, val_recon, val_entropy, val_kl, val_kl_objective, val_z_separation = run_epoch(
                    encoder,
                    decoder,
                    val_loader,
                    optimizer=None,
                    device=device,
                    recon_loss_type=args.recon_loss_type,
                    occupied_weight=args.occupied_weight,
                    focal_gamma=args.focal_gamma,
                    entropy_weight=args.entropy_weight,
                    kl_weight=args.kl_weight,
                    target_kl=args.target_kl,
                    rollout_len=current_rollout_k,
                    teacher_forcing_prob=current_teacher_forcing_p,
                    num_latent_samples=args.num_latent_samples,
                    latent_selection_max_steps=args.latent_selection_max_steps,
                    z_separation_weight=args.z_separation_weight,
                )

            epoch_duration = time.perf_counter() - epoch_start
            total_epoch_time += epoch_duration
            avg_epoch_duration = total_epoch_time / epoch
            epochs_left = args.epochs - epoch
            eta_seconds = avg_epoch_duration * epochs_left

            _log_message(
                f"Epoch {epoch:03d} | "
                f"K={current_rollout_k}, tf_p={current_teacher_forcing_p:.3f}, "
                f"n_z={args.num_latent_samples}, m={args.latent_selection_max_steps} | "
                f"train: loss={train_loss:.6f}, recon={train_recon:.6f}, entropy={train_entropy:.6f}, kl={train_kl:.6f}, kl_obj={train_kl_objective:.6f}, z_sep={train_z_separation:.6f} | "
                f"val: loss={val_loss:.6f}, recon={val_recon:.6f}, entropy={val_entropy:.6f}, kl={val_kl:.6f}, kl_obj={val_kl_objective:.6f}, z_sep={val_z_separation:.6f} | "
                f"epoch_time={_format_hhmmss(epoch_duration)}, eta={_format_hhmmss(eta_seconds)}",
                wandb_run=wandb_run,
            )

            if epoch % args.save_interval == 0:
                periodic_ckpt = args.output.parent / f"{args.output.stem}_periodic_epoch_{epoch:03d}{args.output.suffix}"
                periodic_checkpoint = build_checkpoint(
                    encoder=encoder,
                    decoder=decoder,
                    args=args,
                    curriculum_epochs=curriculum_epochs,
                    decoder_context_latent_channel=decoder_context_latent_channel,
                    downsample_strides=downsample_strides,
                    upsample_strides=upsample_strides,
                    upsample_channels=upsample_channels,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_kl=train_kl,
                    val_kl=val_kl,
                )
                torch.save(periodic_checkpoint, periodic_ckpt)
                _log_message(
                    f"Periodic checkpoint saved: {periodic_ckpt}",
                    wandb_run=wandb_run,
                )

                if wandb_run is not None:
                    # Keep periodic snapshots visible in W&B Files.
                    wandb_run.save(str(periodic_ckpt), base_path=str(args.output.parent), policy="now")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/recon": train_recon,
                        "train/entropy": train_entropy,
                        "train/kl": train_kl,
                        "train/kl_objective": train_kl_objective,
                        "train/z_separation": train_z_separation,
                        "val/loss": val_loss,
                        "val/recon": val_recon,
                        "val/entropy": val_entropy,
                        "val/kl": val_kl,
                        "val/kl_objective": val_kl_objective,
                        "val/z_separation": val_z_separation,
                        "loss/entropy_weight": args.entropy_weight,
                        "loss/z_separation_weight": args.z_separation_weight,
                        "kl/target": args.target_kl,
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

                epoch_ckpt = args.output.parent / f"{args.output.stem}_epoch_{epoch:03d}{args.output.suffix}"
                checkpoint = build_checkpoint(
                    encoder=encoder,
                    decoder=decoder,
                    args=args,
                    curriculum_epochs=curriculum_epochs,
                    decoder_context_latent_channel=decoder_context_latent_channel,
                    downsample_strides=downsample_strides,
                    upsample_strides=upsample_strides,
                    upsample_channels=upsample_channels,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_kl=train_kl,
                    val_kl=val_kl,
                )
                torch.save(checkpoint, epoch_ckpt)
                torch.save(checkpoint, args.output)
                _log_message(
                    f"Validation improved to {best_val_loss:.6f}; "
                    f"saved checkpoint: {epoch_ckpt}",
                    wandb_run=wandb_run,
                )

                if wandb_run is not None:
                    # Save checkpoint files to the run's Files tab for easier discoverability.
                    wandb_run.save(str(epoch_ckpt), base_path=str(args.output.parent), policy="now")
                    wandb_run.save(str(args.output), base_path=str(args.output.parent), policy="now")

                    # Also track them as versioned model artifacts.
                    artifact = wandb.Artifact(
                        name=f"vae-checkpoint-{epoch}",
                        type="model",
                        metadata={"epoch": epoch, "best_val_loss": best_val_loss},
                    )
                    artifact.add_file(str(epoch_ckpt), name=epoch_ckpt.name)
                    artifact.add_file(str(args.output), name=args.output.name)
                    wandb_run.log_artifact(
                        artifact,
                        aliases=["best", "latest", f"epoch-{epoch:03d}"],
                    )
            else:
                _log_message(
                    f"No validation improvement (best={best_val_loss:.6f} at epoch {best_epoch:03d}); "
                    "checkpoint not updated.",
                    wandb_run=wandb_run,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
