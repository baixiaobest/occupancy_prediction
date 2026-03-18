from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.VAE_prediction import VAEPredictionDecoder, VAEPredictionEncoder
from src.rollout_data import RollOutData


@dataclass
class DatasetStats:
    num_scene_files: int
    num_train_origins: int
    num_val_origins: int
    num_train_samples: int
    num_val_samples: int


class OccupancyWindowDataset(Dataset):
    """Sliding-window dataset over occupancy sequences.

    Each sample uses first `history_len` frames as input and predicts the next frame.
    Input shape:  (1, history_len, H, W)
    Target shape: (1, 1, H, W)
    """

    def __init__(self, sequences: Sequence[torch.Tensor], history_len: int = 16, stride: int = 1) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        """Create a sliding-window dataset from a list of occupancy sequences.

        Args:
            sequences: Iterable of tensors with shape `(T, H, W)` representing
                binary occupancy frames for one origin.
            history_len: Number of past frames used as input (the target is the
                next single frame).
            stride: Step size between window starts when creating samples.

        Raises:
            ValueError: If `history_len` or `stride` are non-positive or if
                any sequence does not have shape `(T, H, W)` during indexing.
        """

        self.history_len = int(history_len)
        self.window_size = self.history_len + 1
        self.sequences = list(sequences)
        self.samples: list[tuple[int, int]] = []

        for seq_idx, seq in enumerate(self.sequences):
            if seq.ndim != 3:
                raise ValueError("Each sequence must have shape (T, H, W)")
            t = seq.shape[0]
            if t < self.window_size:
                continue
            for start in range(0, t - self.window_size + 1, stride):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        """Return the number of sliding-window samples available."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (input, target) pair for the sample at `index`.

        Returns:
            A tuple `(x, y)` where `x` has shape `(1, history_len, H, W)` and
            `y` has shape `(1, 1, H, W)`.
        """
        seq_idx, start = self.samples[index]
        seq = self.sequences[seq_idx]

        past = seq[start : start + self.history_len]  # (history_len, H, W)
        future = seq[start + self.history_len : start + self.window_size]  # (1, H, W)

        x = past.unsqueeze(0)  # (1, history_len, H, W)
        y = future.unsqueeze(0)  # (1, 1, H, W)
        return x, y


def _load_scene_origins(pt_path: Path) -> list[torch.Tensor]:
    """Load occupancy sequences from a .pt rollout file.

    Supports loading `RollOutData` instances, dict payloads containing the
    key `"occupancy_grids"`, or plain lists of per-origin frame sequences.

    Args:
        pt_path: Path to the .pt file to load.

    Returns:
        A list of tensors where each tensor has shape `(T, H, W)` and contains
        binary values in {0., 1.}.

    Raises:
        ValueError: If the payload format is not recognized or if frames are
            not 2D arrays.
    """
    payload = torch.load(pt_path, map_location="cpu")

    # Normalize payload into an iterable of per-origin frame lists.
    if isinstance(payload, RollOutData):
        occupancy_items = payload.occupancy_grids
    elif isinstance(payload, dict) and "occupancy_grids" in payload:
        occupancy_items = payload["occupancy_grids"]
    elif isinstance(payload, list):
        # The file may contain a list of origin-series or a list of RollOutData
        # objects (e.g., multiple scenes). If it's a list of RollOutData, extract
        # their `occupancy_grids` and flatten; otherwise treat the list as the
        # occupancy container.
        if all(isinstance(x, RollOutData) for x in payload):
            occupancy_items = []
            for r in payload:
                occupancy_items.extend(r.occupancy_grids)
        else:
            occupancy_items = payload
    else:
        raise ValueError(f"Unsupported payload format in {pt_path}")

    sequences: list[torch.Tensor] = []
    for origin_series in occupancy_items:
        frames: list[torch.Tensor] = []
        for frame in origin_series:
            tensor = torch.as_tensor(frame, dtype=torch.float32)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 2:
                raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
            frames.append(tensor)

        if not frames:
            continue

        seq = torch.stack(frames, dim=0)  # (T, H, W)
        seq = (seq > 0).float()  # ensure binary {0,1}
        sequences.append(seq)

    return sequences


def _split_origins(
    sequences: Sequence[torch.Tensor],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split a sequence list into train and validation subsets.

    The split is performed at the origin (sequence) level using a provided
    RNG for deterministic shuffling when seeded.

    Args:
        sequences: Sequence of tensors `(T, H, W)` to split.
        val_ratio: Fraction of origins to assign to validation (in [0, 1)).
        rng: Random instance for shuffling.

    Returns:
        A pair `(train_sequences, val_sequences)`.

    Raises:
        ValueError: If `val_ratio` is outside the allowed range.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    indices = list(range(len(sequences)))
    rng.shuffle(indices)

    if len(indices) <= 1 or val_ratio == 0.0:
        train_idx = indices
        val_idx: list[int] = []
    else:
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    return train_sequences, val_sequences


def build_datasets(
    data_dir: Path,
    val_ratio: float,
    history_len: int,
    window_stride: int,
    seed: int,
) -> tuple[OccupancyWindowDataset, OccupancyWindowDataset, DatasetStats]:
    """Discover .pt files in `data_dir` and build train/val datasets.

    This function loads each .pt file's per-origin sequences, performs an
    origin-level train/validation split, and returns `OccupancyWindowDataset`
    instances for training and validation along with summary statistics.

    Args:
        data_dir: Directory containing `.pt` rollouts.
        val_ratio: Fraction of origins allocated to validation.
        history_len: Number of past frames used as encoder input.
        window_stride: Sliding-window stride when creating samples.
        seed: RNG seed for deterministic splits.

    Returns:
        `(train_dataset, val_dataset, stats)`.
    """
    rng = random.Random(seed)

    all_train_sequences: list[torch.Tensor] = []
    all_val_sequences: list[torch.Tensor] = []

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    for pt_file in pt_files:
        scene_sequences = _load_scene_origins(pt_file)
        train_seq, val_seq = _split_origins(scene_sequences, val_ratio, rng)
        all_train_sequences.extend(train_seq)
        all_val_sequences.extend(val_seq)

    train_dataset = OccupancyWindowDataset(all_train_sequences, history_len=history_len, stride=window_stride)
    val_dataset = OccupancyWindowDataset(all_val_sequences, history_len=history_len, stride=window_stride)

    stats = DatasetStats(
        num_scene_files=len(pt_files),
        num_train_origins=len(all_train_sequences),
        num_val_origins=len(all_val_sequences),
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )
    return train_dataset, val_dataset, stats


def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute analytic KL divergence between N(mu, sigma^2) and N(0, I).

    Args:
        mu: Tensor of shape `(B, C, T, H, W)` representing latent means.
        sigma: Tensor of the same shape representing standard deviations.

    Returns:
        Scalar tensor: mean KL divergence across the batch.
    """
    var = sigma.pow(2)
    kl = 0.5 * (mu.pow(2) + var - torch.log(var + 1e-8) - 1)
    kl = kl.sum(dim=(1,2,3,4)).mean()

    return kl


def weighted_bernoulli_recon_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    occupied_weight: float,
) -> torch.Tensor:
    """Compute weighted binary cross-entropy between `logits` and `target`.

    Cells where `target == 1` receive `occupied_weight` to emphasize occupied
    cells; other cells get weight 1.0.

    Args:
        logits: Raw model outputs (before sigmoid), shape `(B, C, 1, H, W)`.
        target: Ground-truth binary targets with same spatial shape.
        occupied_weight: Weight multiplier for positive (occupied) cells.

    Returns:
        Scalar loss tensor averaged over the batch.
    """
    if occupied_weight < 1.0:
        raise ValueError("occupied_weight must be >= 1.0")
    pos_w = torch.tensor(float(occupied_weight), device=target.device, dtype=target.dtype)
    weights = torch.where(target > 0.5, pos_w, torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction="mean")


def run_epoch(
    encoder: VAEPredictionEncoder,
    decoder: VAEPredictionDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    occupied_weight: float,
    kl_weight: float,
) -> tuple[float, float, float]:
    """Run one epoch (training or evaluation) over `loader`.

    If `optimizer` is provided the model parameters are updated (training
    mode). Otherwise the models are run in evaluation mode and gradients are
    not applied.

    Args:
        encoder: VAE encoder instance.
        decoder: VAE decoder instance.
        loader: DataLoader yielding `(x, y)` pairs.
        optimizer: Optimizer to use for training, or `None` for evaluation.
        device: Device to run tensors on.
        occupied_weight: Weighting used in reconstruction loss for occupied cells.
        kl_weight: Weight to scale the KL term when computing total loss.

    Returns:
        A tuple `(avg_loss, avg_recon, avg_kl)` averaged over batches.
    """
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        mu, sigma = encoder(x)
        z = encoder.sample(mu, sigma)
        logits = decoder(z)

        recon = weighted_bernoulli_recon_loss(logits, y, occupied_weight=occupied_weight)
        kl = kl_divergence(mu, sigma)
        loss = recon + kl_weight * kl

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kl.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / total_batches,
        total_recon / total_batches,
        total_kl / total_batches,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed argparse namespace with training and model hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train occupancy prediction VAE")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing rollout .pt files")
    parser.add_argument("--history-len", type=int, default=16, help="Number of past frames as encoder input")
    parser.add_argument("--window-size", type=int, default=17, help="Total window length (history + target)")
    parser.add_argument("--window-stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for origin-level split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--occupied-weight", type=float, default=5.0, help="Weight for cells where target=1")
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--latent-channel", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/vae_prediction.pt"))
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Weights & Biases logging",
    )
    parser.add_argument("--wandb-project", type=str, default="occupancy-prediction")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point for training the VAE model.

    Performs dataset discovery, constructs the VAE encoder/decoder, runs the
    training loop for the specified number of epochs, and saves the best
    checkpoint to `args.output`.
    """
    args = parse_args()

    if args.window_size != args.history_len + 1:
        raise ValueError("window_size must be history_len + 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, stats = build_datasets(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        history_len=args.history_len,
        window_stride=args.window_stride,
        seed=args.seed,
    )

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

    sample_x, sample_y = train_dataset[0]
    _, hist_t, h, w = sample_x.shape
    input_shape = (1, hist_t, h, w)
    output_shape = (1, 1, h, w)

    encoder = VAEPredictionEncoder(
        input_shape=input_shape,
        latent_channel=args.latent_channel,
        base_channels=args.base_channels,
        downsample_strides=[(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
    ).to(device)
    decoder = VAEPredictionDecoder(
        latent_dim=args.latent_channel,
        output_shape=output_shape,
        upsample_strides=[(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        upsample_channels=(128, 64, 32, 16, 8, 4, 2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        "Dataset summary: "
        f"files={stats.num_scene_files}, "
        f"train_origins={stats.num_train_origins}, val_origins={stats.num_val_origins}, "
        f"train_samples={stats.num_train_samples}, val_samples={stats.num_val_samples}"
    )

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

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_recon, train_kl = run_epoch(
                encoder,
                decoder,
                train_loader,
                optimizer,
                device,
                occupied_weight=args.occupied_weight,
                kl_weight=args.kl_weight,
            )

            with torch.no_grad():
                val_loss, val_recon, val_kl = run_epoch(
                    encoder,
                    decoder,
                    val_loader,
                    optimizer=None,
                    device=device,
                    occupied_weight=args.occupied_weight,
                    kl_weight=args.kl_weight,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"train: loss={train_loss:.6f}, recon={train_recon:.6f}, kl={train_kl:.6f} | "
                f"val: loss={val_loss:.6f}, recon={val_recon:.6f}, kl={val_kl:.6f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/recon": train_recon,
                        "train/kl": train_kl,
                        "val/loss": val_loss,
                        "val/recon": val_recon,
                        "val/kl": val_kl,
                    },
                    step=epoch,
                    commit=True,
                )

            # Save checkpoint 
            epoch_ckpt = args.output.parent / f"{args.output.stem}_epoch_{epoch:03d}{args.output.suffix}"
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "args": vars(args),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_kl": train_kl,
                "val_kl": val_kl,
            }
            torch.save(checkpoint, epoch_ckpt)
            torch.save(checkpoint, args.output)
            print(f"Saved checkpoint: {epoch_ckpt}")

            if wandb_run is not None:
                # Save checkpoint files to the run's Files tab for easier discoverability.
                wandb_run.save(str(epoch_ckpt), base_path=str(args.output.parent), policy="now")
                wandb_run.save(str(args.output), base_path=str(args.output.parent), policy="now")

                # Also track them as versioned model artifacts.
                artifact = wandb.Artifact(
                    name=f"vae-checkpoint-{wandb_run.id}",
                    type="model",
                    metadata={"epoch": epoch},
                )
                artifact.add_file(str(epoch_ckpt), name=epoch_ckpt.name)
                artifact.add_file(str(args.output), name=args.output.name)
                wandb_run.log_artifact(
                    artifact,
                    aliases=["latest", f"epoch-{epoch:03d}"],
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
