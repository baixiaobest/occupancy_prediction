from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.VAE_prediction import VAEPredictionDecoder, VAEPredictionEncoder
from src.rollout_data import RollOutData


def _parse_stride_token(token: str) -> tuple[int, int, int]:
    parts = token.strip().split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid stride token '{token}'. Expected format like 2x2x2")
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_stride_list(text: str) -> list[tuple[int, int, int]]:
    return [_parse_stride_token(tok) for tok in text.split(",") if tok.strip()]


def parse_channel_list(text: str) -> list[int]:
    channels = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not channels:
        raise ValueError("upsample_channels cannot be empty")
    return channels


def load_scene_origins(pt_path: Path) -> list[torch.Tensor]:
    """Load occupancy time-series per scene/origin from a .pt file.

    Returns list of tensors with shape (T, H, W), binary in {0,1}.
    """
    payload = torch.load(pt_path, map_location="cpu")

    if isinstance(payload, RollOutData):
        occupancy_items = payload.occupancy_grids
    elif isinstance(payload, dict) and "occupancy_grids" in payload:
        occupancy_items = payload["occupancy_grids"]
    elif isinstance(payload, list):
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
            frames.append((tensor > 0).float())

        if frames:
            sequences.append(torch.stack(frames, dim=0))

    return sequences


def build_models(
    checkpoint_path: Path,
    device: torch.device,
    fallback_input_shape: Sequence[int],
    fallback_output_shape: Sequence[int],
    downsample_strides: list[tuple[int, int, int]],
    upsample_strides: list[tuple[int, int, int]],
    upsample_channels: list[int],
) -> tuple[VAEPredictionEncoder, VAEPredictionDecoder]:
    """Construct encoder/decoder and load checkpoint weights."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    latent_channel = int(args.get("latent_channel", upsample_channels[0]))
    base_channels = int(args.get("base_channels", 32))

    input_shape = tuple(ckpt.get("input_shape", fallback_input_shape))
    output_shape = tuple(ckpt.get("output_shape", fallback_output_shape))

    encoder = VAEPredictionEncoder(
        input_shape=input_shape,
        latent_channel=latent_channel,
        base_channels=base_channels,
        downsample_strides=downsample_strides,
    ).to(device)

    decoder = VAEPredictionDecoder(
        latent_dim=latent_channel,
        output_shape=output_shape,
        upsample_channels=upsample_channels,
        upsample_strides=upsample_strides,
    ).to(device)

    if not isinstance(ckpt, dict) or "encoder" not in ckpt or "decoder" not in ckpt:
        raise ValueError("Checkpoint must contain keys: 'encoder' and 'decoder'")

    encoder.load_state_dict(ckpt["encoder"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def autoregressive_predict(
    sequence: torch.Tensor,
    encoder: VAEPredictionEncoder,
    decoder: VAEPredictionDecoder,
    history_len: int,
    horizon: int,
    device: torch.device,
    binary_feedback: bool,
    threshold: float,
) -> torch.Tensor:
    """Compute predictions for all anchors t in [history_len, T-1].

    Returns tensor of shape (T-history_len, horizon, H, W).
    """
    t_total, h, w = sequence.shape
    if t_total <= history_len:
        return torch.zeros((0, horizon, h, w), dtype=torch.float32)

    all_preds = torch.zeros((t_total - history_len, horizon, h, w), dtype=torch.float32)

    with torch.no_grad():
        for anchor_t in range(history_len, t_total):
            context = sequence[anchor_t - history_len : anchor_t].clone()  # (history_len, H, W)
            pred_frames: list[torch.Tensor] = []
            for _ in range(horizon):
                x = context[-history_len:].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,history,H,W)
                mu, _ = encoder(x)
                logits = decoder(mu)
                prob = torch.sigmoid(logits)[0, 0, 0].detach().cpu()
                pred_frames.append(prob)

                feedback = (prob >= threshold).float() if binary_feedback else prob
                context = torch.cat([context, feedback.unsqueeze(0)], dim=0)

            all_preds[anchor_t - history_len] = torch.stack(pred_frames, dim=0)

    return all_preds


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive visualizer for VAE occupancy predictions")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to rollout .pt file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--origin-index", "--scene-index", dest="origin_index", type=int, default=0, help="Initial origin-center index")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4, help="Initial autoregressive horizon")
    parser.add_argument("--max-horizon", type=int, default=16, help="Max horizon in GUI slider")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--downsample-strides",
        type=str,
        default="2x2x2,2x2x2,1x2x2,1x2x2,1x2x2,1x2x2",
        help="Comma-separated 3D strides, e.g. 4x4x4,1x4x4,1x4x4",
    )
    parser.add_argument(
        "--upsample-strides",
        type=str,
        default="2x2x2,2x2x2,1x2x2,1x2x2,1x2x2,1x2x2",
        help="Comma-separated 3D strides, e.g. 4x4x4,1x4x4,1x4x4",
    )
    parser.add_argument(
        "--upsample-channels",
        type=str,
        default="128,64,32,16,8,4,2",
        help="Comma-separated channels [input, block1_out, ..., blockN_out]",
    )
    parser.add_argument("--binary-feedback", action="store_true", help="Feed thresholded predictions back into context")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary feedback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    downsample_strides = parse_stride_list(args.downsample_strides)
    upsample_strides = parse_stride_list(args.upsample_strides)
    upsample_channels = parse_channel_list(args.upsample_channels)

    sequences = load_scene_origins(args.data_file)
    if not sequences:
        raise ValueError(f"No scene sequences found in {args.data_file}")

    for i, seq in enumerate(sequences):
        if seq.ndim != 3:
            raise ValueError(f"Origin {i} has invalid shape {tuple(seq.shape)}; expected (T,H,W)")

    init_origin = clamp_int(args.origin_index, 0, len(sequences) - 1)
    init_seq = sequences[init_origin]
    _, init_h, init_w = init_seq.shape

    # Fallback shapes are used when checkpoint metadata does not provide them.
    fallback_input_shape = (1, args.history_len, init_h, init_w)
    fallback_output_shape = (1, 1, init_h, init_w)

    encoder, decoder = build_models(
        checkpoint_path=args.checkpoint,
        device=device,
        fallback_input_shape=fallback_input_shape,
        fallback_output_shape=fallback_output_shape,
        downsample_strides=downsample_strides,
        upsample_strides=upsample_strides,
        upsample_channels=upsample_channels,
    )

    # Cache by (origin_index, horizon) to avoid recomputing when only t changes.
    cache: dict[tuple[int, int], torch.Tensor] = {}

    def get_predictions(origin_idx: int, horizon: int) -> torch.Tensor:
        key = (origin_idx, horizon)
        if key not in cache:
            print(f"Running inference for origin={origin_idx}, horizon={horizon} ...")
            seq = sequences[origin_idx]
            cache[key] = autoregressive_predict(
                sequence=seq,
                encoder=encoder,
                decoder=decoder,
                history_len=args.history_len,
                horizon=horizon,
                device=device,
                binary_feedback=args.binary_feedback,
                threshold=args.threshold,
            )
            t_total = seq.shape[0]
            print(f"Done. Total inferences: {(max(t_total - args.history_len, 0)) * horizon}")
        return cache[key]

    # Build figure and axes.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_past, ax_pred, ax_overlay_pred, ax_overlay_gt = axes.flatten()
    plt.subplots_adjust(bottom=0.22)

    zero_img = np.zeros((init_h, init_w), dtype=np.float32)
    im_past = ax_past.imshow(zero_img, cmap="Blues", vmin=0.0, vmax=1.0)
    im_pred = ax_pred.imshow(zero_img, cmap="Reds", vmin=0.0, vmax=1.0)
    im_overlay_pred = ax_overlay_pred.imshow(np.zeros((init_h, init_w, 3), dtype=np.float32), vmin=0.0, vmax=1.0)
    im_overlay_gt = ax_overlay_gt.imshow(np.zeros((init_h, init_w, 3), dtype=np.float32), vmin=0.0, vmax=1.0)

    ax_past.set_title("Past 16 Stack")
    ax_pred.set_title("Predicted Horizon Stack")
    ax_overlay_pred.set_title("Overlay: Past (Blue) + Pred (Red)")
    ax_overlay_gt.set_title("Overlay: Past (Blue) + GT Future (Green)")

    for ax in [ax_past, ax_pred, ax_overlay_pred, ax_overlay_gt]:
        ax.set_xticks([])
        ax.set_yticks([])

    # Sliders.
    slider_color = "lightgoldenrodyellow"
    ax_origin = plt.axes([0.12, 0.14, 0.78, 0.03], facecolor=slider_color)
    ax_t = plt.axes([0.12, 0.09, 0.78, 0.03], facecolor=slider_color)
    ax_h = plt.axes([0.12, 0.04, 0.78, 0.03], facecolor=slider_color)

    origin_slider = Slider(ax_origin, "Origin", 0, max(0, len(sequences) - 1), valinit=init_origin, valstep=1)

    max_t_global = max(seq.shape[0] - 1 for seq in sequences)
    t_slider = Slider(ax_t, "t", args.history_len, max_t_global, valinit=max(args.history_len, args.history_len), valstep=1)

    h_slider = Slider(
        ax_h,
        "Horizon",
        1,
        max(1, args.max_horizon),
        valinit=clamp_int(args.horizon, 1, max(1, args.max_horizon)),
        valstep=1,
    )

    info_text = fig.text(0.02, 0.96, "", fontsize=10, va="top")

    def refresh(_: float | None = None) -> None:
        origin_idx = int(origin_slider.val)
        horizon = int(h_slider.val)

        seq = sequences[origin_idx]
        t_total, h_img, w_img = seq.shape
        t_min = args.history_len
        t_max = max(t_min, t_total - 1)
        t_req = int(t_slider.val)
        t = clamp_int(t_req, t_min, t_max)

        if t != t_req:
            t_slider.set_val(t)
            return

        preds = get_predictions(origin_idx, horizon)

        past = seq[t - args.history_len : t]  # (history_len, H, W)
        past_stack = past.max(dim=0).values

        if preds.shape[0] > 0:
            pred_index = t - args.history_len
            pred_seq = preds[pred_index]  # (horizon, H, W)
            pred_stack = pred_seq.max(dim=0).values
        else:
            pred_stack = torch.zeros((h_img, w_img), dtype=torch.float32)

        gt_future = seq[t : min(t + horizon, t_total)]
        if gt_future.shape[0] > 0:
            gt_stack = gt_future.max(dim=0).values
        else:
            gt_stack = torch.zeros((h_img, w_img), dtype=torch.float32)

        past_np = past_stack.numpy()
        pred_np = pred_stack.numpy()
        gt_np = gt_stack.numpy()

        overlay_pred = np.stack([pred_np, np.zeros_like(pred_np), past_np], axis=-1)
        overlay_gt = np.stack([np.zeros_like(gt_np), gt_np, past_np], axis=-1)

        im_past.set_data(past_np)
        im_pred.set_data(pred_np)
        im_overlay_pred.set_data(np.clip(overlay_pred, 0.0, 1.0))
        im_overlay_gt.set_data(np.clip(overlay_gt, 0.0, 1.0))

        ax_past.set_title(f"Past Stack (t-{args.history_len}..t-1), t={t}")
        ax_pred.set_title(f"Pred Stack (h={horizon})")

        total_infer = max(t_total - args.history_len, 0) * horizon
        info_text.set_text(
            f"origin={origin_idx} | T={t_total} | t={t} | horizon={horizon} | total inferences={(total_infer)}"
        )

        fig.canvas.draw_idle()

    origin_slider.on_changed(refresh)
    t_slider.on_changed(refresh)
    h_slider.on_changed(refresh)

    refresh(None)
    plt.show()


if __name__ == "__main__":
    main()
