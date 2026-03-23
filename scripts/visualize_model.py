from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import RadioButtons, Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.VAE_prediction import (
    VAEPredictionDecoder,
    build_prediction_vae_models,
)
from src.rollout_data import RollOutData

def _coerce_stride_list(raw: object) -> list[tuple[int, int, int]]:
    if raw is None:
        raise ValueError("stride config is missing")
    return [
        (int(s[0]), int(s[1]), int(s[2]))
        for s in raw
    ]


def _coerce_channel_list(raw: object) -> list[int]:
    if raw is None:
        raise ValueError("channel config is missing")
    return [int(c) for c in raw]


def load_scene_origins(pt_path: Path) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Load dynamic time-series and static maps grouped by scene and origin.

    Returns:
        scene_dynamic_sequences: list[scene] of list[origin] tensors `(T, H, W)`
        scene_static_maps: list[scene] of list[origin] tensors `(H, W)`
    """
    payload = torch.load(pt_path, map_location="cpu")

    def _to_2d(frame: object) -> torch.Tensor:
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
        return (tensor > 0).float()

    scene_dynamic_sequences: list[list[torch.Tensor]] = []
    scene_static_maps: list[list[torch.Tensor]] = []

    def _collect_from_rollout_obj(obj: RollOutData) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        dynamic_sequences: list[torch.Tensor] = []
        static_maps: list[torch.Tensor] = []
        if hasattr(obj, "dynamic_grids") and hasattr(obj, "static_maps"):
            if len(obj.dynamic_grids) != len(obj.static_maps):
                raise ValueError("dynamic_grids and static_maps must have same length")
            for dyn_series, static_map in zip(obj.dynamic_grids, obj.static_maps):
                dyn_frames = [_to_2d(frame) for frame in dyn_series]
                if not dyn_frames:
                    continue
                dynamic_sequences.append(torch.stack(dyn_frames, dim=0))
                static_maps.append(_to_2d(static_map))
            return dynamic_sequences, static_maps

        if hasattr(obj, "occupancy_grids"):
            for origin_series in obj.occupancy_grids:
                frames = [_to_2d(frame) for frame in origin_series]
                if not frames:
                    continue
                full_seq = torch.stack(frames, dim=0)
                dynamic_sequences.append(full_seq)
                static_maps.append(torch.zeros_like(full_seq[0]))
            return dynamic_sequences, static_maps

        raise ValueError("Unsupported RollOutData schema")

    if isinstance(payload, RollOutData):
        dyn, sta = _collect_from_rollout_obj(payload)
        if dyn:
            scene_dynamic_sequences.append(dyn)
            scene_static_maps.append(sta)
    elif isinstance(payload, dict):
        if "dynamic_grids" in payload and "static_maps" in payload:
            pseudo = RollOutData(
                static_maps=payload["static_maps"],
                dynamic_grids=payload["dynamic_grids"],
                dt=float(payload.get("dt", 0.0)),
            )
            dyn, sta = _collect_from_rollout_obj(pseudo)
            if dyn:
                scene_dynamic_sequences.append(dyn)
                scene_static_maps.append(sta)
        elif "occupancy_grids" in payload:
            pseudo = RollOutData(static_maps=[], dynamic_grids=[], dt=float(payload.get("dt", 0.0)))
            setattr(pseudo, "occupancy_grids", payload["occupancy_grids"])
            dyn, sta = _collect_from_rollout_obj(pseudo)
            if dyn:
                scene_dynamic_sequences.append(dyn)
                scene_static_maps.append(sta)
        else:
            raise ValueError(f"Unsupported payload format in {pt_path}")
    elif isinstance(payload, list):
        if all(isinstance(x, RollOutData) for x in payload):
            for item in payload:
                dyn, sta = _collect_from_rollout_obj(item)
                if dyn:
                    scene_dynamic_sequences.append(dyn)
                    scene_static_maps.append(sta)
        else:
            # Treat list-of-origins payload as a single scene with many origins.
            dynamic_sequences: list[torch.Tensor] = []
            static_maps: list[torch.Tensor] = []
            for origin_series in payload:
                frames = [_to_2d(frame) for frame in origin_series]
                if not frames:
                    continue
                full_seq = torch.stack(frames, dim=0)
                dynamic_sequences.append(full_seq)
                static_maps.append(torch.zeros_like(full_seq[0]))
            if dynamic_sequences:
                scene_dynamic_sequences.append(dynamic_sequences)
                scene_static_maps.append(static_maps)
    else:
        raise ValueError(f"Unsupported payload format in {pt_path}")

    return scene_dynamic_sequences, scene_static_maps


def build_models(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VAEPredictionDecoder, int, int, tuple[int, int, int], int]:
    """Construct decoder from checkpoint config and load checkpoint weights."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict with model config and state dicts")
    model_cfg = ckpt.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError("Checkpoint must contain dict key 'model_config'")

    required_keys = [
        "history_len",
        "latent_channel",
        "base_channels",
        "static_stem_channels",
        "input_shape",
        "output_shape",
        "downsample_strides",
        "upsample_strides",
        "upsample_channels",
    ]
    missing = [key for key in required_keys if key not in model_cfg]
    if missing:
        raise ValueError(f"Checkpoint model_config is missing required keys: {missing}")

    history_len = int(model_cfg["history_len"])
    decoder_context_len = int(model_cfg.get("decoder_context_len", min(8, history_len)))
    latent_channel = int(model_cfg["latent_channel"])
    base_channels = int(model_cfg["base_channels"])
    decoder_base_channels = int(model_cfg.get("decoder_base_channels", base_channels))
    decoder_context_latent_channel = int(
        model_cfg.get("decoder_context_latent_channel", latent_channel)
    )
    static_stem_channels = int(model_cfg["static_stem_channels"])

    input_shape = tuple(model_cfg["input_shape"])
    output_shape = tuple(model_cfg["output_shape"])
    model_downsample_strides = _coerce_stride_list(model_cfg["downsample_strides"])
    model_upsample_strides = _coerce_stride_list(model_cfg["upsample_strides"])
    model_upsample_channels = _coerce_channel_list(model_cfg["upsample_channels"])

    if any(v <= 0 for v in input_shape) or any(v <= 0 for v in output_shape):
        raise ValueError("Checkpoint is missing valid input_shape/output_shape for model reconstruction")

    _encoder, decoder = build_prediction_vae_models(
        input_shape=input_shape,
        output_shape=output_shape,
        latent_channel=latent_channel,
        base_channels=base_channels,
        decoder_base_channels=decoder_base_channels,
        decoder_context_latent_channel=decoder_context_latent_channel,
        static_stem_channels=static_stem_channels,
        downsample_strides=model_downsample_strides,
        upsample_strides=model_upsample_strides,
        upsample_channels=model_upsample_channels,
        device=device,
    )

    if "decoder" not in ckpt:
        raise ValueError("Checkpoint must contain key: 'decoder'")

    decoder.load_state_dict(ckpt["decoder"], strict=True)

    decoder.eval()

    latent_t, latent_h, latent_w = input_shape[1], input_shape[2], input_shape[3]
    for stride in model_downsample_strides:
        latent_t = (latent_t + stride[0] - 1) // stride[0]
        latent_h = (latent_h + stride[1] - 1) // stride[1]
        latent_w = (latent_w + stride[2] - 1) // stride[2]

    return decoder, history_len, decoder_context_len, (latent_t, latent_h, latent_w), latent_channel


def autoregressive_predict(
    dynamic_sequence: torch.Tensor,
    static_map: torch.Tensor,
    decoder: VAEPredictionDecoder,
    history_len: int,
    decoder_context_len: int,
    latent_shape: tuple[int, int, int],
    latent_channels: int,
    horizon: int,
    device: torch.device,
    binary_feedback: bool,
    threshold: float,
) -> torch.Tensor:
    """Compute predictions for all anchors t in [history_len, T-1].

    Returns tensor of shape (T-history_len, horizon, H, W).
    """
    t_total, h, w = dynamic_sequence.shape
    if t_total <= history_len:
        return torch.zeros((0, horizon, h, w), dtype=torch.float32)

    all_preds = torch.zeros((t_total - history_len, horizon, h, w), dtype=torch.float32)

    with torch.no_grad():
        for anchor_t in range(history_len, t_total):
            context = dynamic_sequence[anchor_t - history_len : anchor_t].clone()  # (history_len, H, W)
            z = torch.randn((1, latent_channels, latent_shape[0], latent_shape[1], latent_shape[2]), device=device)
            pred_frames: list[torch.Tensor] = []
            for _ in range(horizon):
                x_decoder_dynamic = context[-decoder_context_len:].unsqueeze(0).unsqueeze(0).to(device)
                x_static = static_map.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
                logits = decoder(z, x_decoder_dynamic, x_static)
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
    parser.add_argument("--scene-index", type=int, default=0, help="Initial scene index")
    parser.add_argument("--origin-index", type=int, default=0, help="Initial origin index within selected scene")
    parser.add_argument("--horizon", type=int, default=4, help="Initial autoregressive horizon")
    parser.add_argument("--max-horizon", type=int, default=16, help="Max horizon in GUI slider")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--binary-feedback", action="store_true", help="Feed thresholded predictions back into context")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary feedback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    scene_sequences, scene_static_maps = load_scene_origins(args.data_file)
    if not scene_sequences:
        raise ValueError(f"No scene sequences found in {args.data_file}")

    for scene_idx, origins in enumerate(scene_sequences):
        if not origins:
            raise ValueError(f"Scene {scene_idx} has no origins")
        for origin_idx, seq in enumerate(origins):
            if seq.ndim != 3:
                raise ValueError(
                    f"Scene {scene_idx}, origin {origin_idx} has invalid shape {tuple(seq.shape)}; expected (T,H,W)"
                )

    init_scene = clamp_int(args.scene_index, 0, len(scene_sequences) - 1)
    init_origin = clamp_int(args.origin_index, 0, len(scene_sequences[init_scene]) - 1)
    init_seq = scene_sequences[init_scene][init_origin]
    _, init_h, init_w = init_seq.shape

    decoder, history_len, decoder_context_len, latent_shape, latent_channels = build_models(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Cache by (scene_index, origin_index, horizon) to avoid recomputing when only t changes.
    cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def get_predictions(scene_idx: int, origin_idx: int, horizon: int) -> torch.Tensor:
        key = (scene_idx, origin_idx, horizon)
        if key not in cache:
            print(f"Running inference for scene={scene_idx}, origin={origin_idx}, horizon={horizon} ...")
            seq = scene_sequences[scene_idx][origin_idx]
            static_map = scene_static_maps[scene_idx][origin_idx]
            cache[key] = autoregressive_predict(
                dynamic_sequence=seq,
                static_map=static_map,
                decoder=decoder,
                history_len=history_len,
                decoder_context_len=decoder_context_len,
                latent_shape=latent_shape,
                latent_channels=latent_channels,
                horizon=horizon,
                device=device,
                binary_feedback=args.binary_feedback,
                threshold=args.threshold,
            )
            t_total = seq.shape[0]
            print(f"Done. Total inferences: {(max(t_total - history_len, 0)) * horizon}")
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
    ax_scene = plt.axes([0.12, 0.12, 0.18, 0.10], facecolor=slider_color)
    ax_origin = plt.axes([0.34, 0.17, 0.56, 0.03], facecolor=slider_color)
    ax_t = plt.axes([0.12, 0.09, 0.78, 0.03], facecolor=slider_color)
    ax_h = plt.axes([0.12, 0.04, 0.78, 0.03], facecolor=slider_color)

    scene_options = [str(i) for i in range(len(scene_sequences))]
    scene_radio = RadioButtons(ax_scene, labels=scene_options, active=init_scene)
    ax_scene.set_title("Scene", fontsize=9)
    scene_state = {"index": init_scene}

    origin_slider = Slider(
        ax_origin,
        "Origin",
        0,
        max(0, len(scene_sequences[init_scene]) - 1),
        valinit=init_origin,
        valstep=1,
    )

    max_t_global = max(seq.shape[0] - 1 for origins in scene_sequences for seq in origins)
    t_slider = Slider(ax_t, "t", history_len, max_t_global, valinit=history_len, valstep=1)

    h_slider = Slider(
        ax_h,
        "Horizon",
        1,
        max(1, args.max_horizon),
        valinit=clamp_int(args.horizon, 1, max(1, args.max_horizon)),
        valstep=1,
    )

    info_text = fig.text(0.02, 0.96, "", fontsize=10, va="top")

    def _on_scene_select(selection: str) -> None:
        try:
            scene_state["index"] = int(selection)
        except ValueError:
            return
        refresh(None)

    def refresh(_: float | None = None) -> None:
        scene_idx = int(scene_state["index"])
        origin_max = max(0, len(scene_sequences[scene_idx]) - 1)
        if origin_slider.valmax != origin_max:
            origin_slider.valmax = origin_max
            origin_slider.ax.set_xlim(origin_slider.valmin, origin_max)

        origin_req = int(origin_slider.val)
        origin_idx = clamp_int(origin_req, 0, origin_max)
        if origin_idx != origin_req:
            origin_slider.set_val(origin_idx)
            return

        horizon = int(h_slider.val)

        seq = scene_sequences[scene_idx][origin_idx]
        t_total, h_img, w_img = seq.shape
        t_min = history_len
        t_max = max(t_min, t_total - 1)
        t_req = int(t_slider.val)
        t = clamp_int(t_req, t_min, t_max)

        if t != t_req:
            t_slider.set_val(t)
            return

        preds = get_predictions(scene_idx, origin_idx, horizon)

        past = seq[t - history_len : t]  # (history_len, H, W)
        past_stack = past.max(dim=0).values

        if preds.shape[0] > 0:
            pred_index = t - history_len
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

        ax_past.set_title(f"Past Stack (t-{history_len}..t-1), t={t}")
        ax_pred.set_title(f"Pred Stack (h={horizon})")

        total_infer = max(t_total - history_len, 0) * horizon
        info_text.set_text(
            f"scene={scene_idx} | origin={origin_idx} | T={t_total} | t={t} | horizon={horizon} | total inferences={(total_infer)}"
        )

        fig.canvas.draw_idle()

    scene_radio.on_clicked(_on_scene_select)
    origin_slider.on_changed(refresh)
    t_slider.on_changed(refresh)
    h_slider.on_changed(refresh)

    refresh(None)
    plt.show()


if __name__ == "__main__":
    main()
