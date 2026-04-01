from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.VAE_prediction import (
    VAEPredictionDecoder,
    build_prediction_vae_models,
)
from src.rollout_data import RollOutData, SceneRollOutData

def _coerce_stride_list(raw: object) -> list[tuple[int, int]]:
    if raw is None:
        raise ValueError("stride config is missing")
    stride_list: list[tuple[int, int]] = []
    for s in raw:
        if len(s) != 2:
            raise ValueError("stride entries must have length 2")
        stride_list.append((int(s[0]), int(s[1])))
    return stride_list


def _coerce_channel_list(raw: object) -> list[int]:
    if raw is None:
        raise ValueError("channel config is missing")
    return [int(c) for c in raw]


def load_scene_origins(
    pt_path: Path,
    history_len: int,
    future_len: int,
) -> tuple[
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[list[int]]],
]:
    """Load dynamic time-series and static maps grouped by scene and origin.

    Returns:
        scene_dynamic_sequences: list[scene] of list[agent] tensors `(A, T, H, W)`
        scene_static_maps: list[scene] of list[agent] tensors `(A, T, H, W)`
        scene_velocity_sequences: list[scene] of list[agent] tensors `(A, T, 2)`
        scene_anchor_times: list[scene] of list[agent] anchor-time lists length `A`

    For one selected agent, the time slider indexes anchor slots (A). Each slot
    uses a fixed anchor center to slice a local window, so it is not a moving frame.
    """
    try:
        payload = torch.load(pt_path, map_location="cpu")
    except AttributeError as exc:
        raise ValueError(
            f"Failed to load {pt_path}: legacy rollout format is no longer supported. "
            "Regenerate rollout .pt files using the current scripts/ORCA_rollout.py format."
        ) from exc

    def _to_2d(frame: object) -> torch.Tensor:
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Occupancy frame must be 2D, got shape {tuple(tensor.shape)}")
        return (tensor > 0).float()

    scene_dynamic_sequences: list[list[torch.Tensor]] = []
    scene_static_maps: list[list[torch.Tensor]] = []
    scene_velocity_sequences: list[list[torch.Tensor]] = []
    scene_anchor_times: list[list[list[int]]] = []

    def _slice_centered_patch(
        grid: torch.Tensor,
        center_xy: torch.Tensor,
        origin_xy: tuple[float, float],
        resolution_xy: tuple[float, float],
        patch_shape: tuple[int, int],
    ) -> torch.Tensor:
        patch_h, patch_w = patch_shape
        res_x, res_y = float(resolution_xy[0]), float(resolution_xy[1])
        center_x = float(center_xy[0].item())
        center_y = float(center_xy[1].item())
        half_w = 0.5 * patch_w * res_x
        half_h = 0.5 * patch_h * res_y
        start_x = int(np.floor((center_x - half_w - origin_xy[0]) / res_x))
        start_y = int(np.floor((center_y - half_h - origin_xy[1]) / res_y))

        out = torch.zeros((patch_h, patch_w), dtype=torch.float32)
        src_x0 = max(0, start_x)
        src_y0 = max(0, start_y)
        src_x1 = min(grid.shape[1], start_x + patch_w)
        src_y1 = min(grid.shape[0], start_y + patch_h)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out

        dst_x0 = src_x0 - start_x
        dst_y0 = src_y0 - start_y
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = grid[src_y0:src_y1, src_x0:src_x1]
        return (out > 0).float()

    def _collect_from_scene_obj(
        scene: SceneRollOutData,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[list[int]]]:
        if scene.scene_static_map is None or scene.scene_map_origin is None or scene.local_map_shape is None:
            raise ValueError("Compact RollOutData fields are required")
        if scene.scene_dynamic_maps is None:
            raise ValueError("RollOutData.scene_dynamic_maps is required")

        dynamic_sequences: list[torch.Tensor] = []
        static_maps: list[torch.Tensor] = []
        velocity_sequences: list[torch.Tensor] = []
        anchor_times_per_agent: list[list[int]] = []

        scene_static = _to_2d(scene.scene_static_map)
        scene_origin = (float(scene.scene_map_origin[0]), float(scene.scene_map_origin[1]))
        resolution_xy = (float(scene.occupancy_resolution[0]), float(scene.occupancy_resolution[1]))
        patch_shape = (int(scene.local_map_shape[0]), int(scene.local_map_shape[1]))
        dynamic_maps = torch.as_tensor(scene.scene_dynamic_maps, dtype=torch.float32)
        if dynamic_maps.ndim != 4:
            raise ValueError("scene_dynamic_maps must have shape (num_agents, total_time, H, W)")
        window_size = int(history_len + future_len)
        if window_size <= 0:
            raise ValueError("history_len + future_len must be > 0")

        for agent_idx in sorted(scene.agents.keys()):
            agent_data = scene.agents[agent_idx]
            if agent_idx < 0 or agent_idx >= dynamic_maps.shape[0]:
                continue

            if agent_data.anchor_centers is None or agent_data.current_velocities is None:
                raise ValueError("Agent metadata requires anchor_centers and current_velocities")

            centers = torch.as_tensor(agent_data.anchor_centers, dtype=torch.float32)
            velocities = torch.as_tensor(agent_data.current_velocities, dtype=torch.float32)
            anchor_times = [int(t) for t in agent_data.anchor_times]

            if centers.ndim != 2 or centers.shape[1] != 2:
                raise ValueError("anchor_centers must have shape (A, 2)")
            if velocities.shape != centers.shape:
                raise ValueError("current_velocities must match anchor_centers shape")
            if len(anchor_times) != centers.shape[0]:
                raise ValueError("anchor_times must match anchor_centers length")

            agent_dynamic = dynamic_maps[agent_idx]
            total_time = int(agent_dynamic.shape[0])

            agent_dynamic_windows: list[torch.Tensor] = []
            agent_static_windows: list[torch.Tensor] = []
            agent_velocity_windows: list[torch.Tensor] = []
            agent_anchor_times: list[int] = []

            for anchor_idx, anchor_t in enumerate(anchor_times):
                start_t = int(anchor_t - history_len)
                end_t = int(anchor_t + future_len)
                if start_t < 0 or end_t > total_time:
                    continue

                center_xy = centers[anchor_idx]

                dynamic_window: list[torch.Tensor] = []
                for absolute_t in range(start_t, end_t):
                    dynamic_window.append(
                        _slice_centered_patch(
                            _to_2d(agent_dynamic[absolute_t]),
                            center_xy,
                            scene_origin,
                            resolution_xy,
                            patch_shape,
                        )
                    )

                if len(dynamic_window) != window_size:
                    continue

                static_local = _slice_centered_patch(
                    scene_static,
                    center_xy,
                    scene_origin,
                    resolution_xy,
                    patch_shape,
                )
                static_sequence = torch.stack([static_local.clone() for _ in range(window_size)], dim=0)
                velocity_sequence = velocities[anchor_idx].to(dtype=torch.float32).unsqueeze(0).repeat(window_size, 1)

                agent_dynamic_windows.append(torch.stack(dynamic_window, dim=0))
                agent_static_windows.append(static_sequence)
                agent_velocity_windows.append(velocity_sequence)
                agent_anchor_times.append(int(anchor_t))

            if not agent_dynamic_windows:
                continue

            dynamic_sequences.append(torch.stack(agent_dynamic_windows, dim=0))
            static_maps.append(torch.stack(agent_static_windows, dim=0))
            velocity_sequences.append(torch.stack(agent_velocity_windows, dim=0))
            anchor_times_per_agent.append(agent_anchor_times)

        return dynamic_sequences, static_maps, velocity_sequences, anchor_times_per_agent

    if isinstance(payload, RollOutData):
        for scene in payload.scenes:
            dyn, sta, vel, anchor_times = _collect_from_scene_obj(scene)
            if dyn:
                scene_dynamic_sequences.append(dyn)
                scene_static_maps.append(sta)
                scene_velocity_sequences.append(vel)
                scene_anchor_times.append(anchor_times)
    else:
        raise ValueError(f"Unsupported payload format in {pt_path}: expected RollOutData")

    return scene_dynamic_sequences, scene_static_maps, scene_velocity_sequences, scene_anchor_times


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
    model_downsample_strides = _coerce_stride_list(model_cfg["downsample_strides"])
    channels_cfg = model_cfg.get("channels")
    if channels_cfg is None:
        raise ValueError("Checkpoint model_config is missing required key: channels")
    channels = [int(c) for c in channels_cfg]

    decoder_base_channels = int(model_cfg.get("decoder_base_channels", channels[0]))
    decoder_downsample_channels_cfg = model_cfg.get("decoder_downsample_channels")
    if decoder_downsample_channels_cfg is None:
        raise ValueError("Checkpoint model_config is missing required key: decoder_downsample_channels")
    decoder_downsample_channels = [int(c) for c in decoder_downsample_channels_cfg]
    decoder_context_latent_channel = int(
        model_cfg.get("decoder_context_latent_channel", latent_channel)
    )
    static_stem_channels = int(model_cfg["static_stem_channels"])
    velocity_mlp_dim = int(model_cfg.get("velocity_mlp_dim", 16))
    encoder_velocity_condition_channels = int(model_cfg.get("encoder_velocity_condition_channels", 0))
    decoder_velocity_condition_channels = int(model_cfg.get("decoder_velocity_condition_channels", 0))

    input_shape = tuple(model_cfg["input_shape"])
    output_shape = tuple(model_cfg["output_shape"])
    model_upsample_strides = _coerce_stride_list(model_cfg["upsample_strides"])
    model_upsample_channels = _coerce_channel_list(model_cfg["upsample_channels"])

    if any(v <= 0 for v in input_shape) or any(v <= 0 for v in output_shape):
        raise ValueError("Checkpoint is missing valid input_shape/output_shape for model reconstruction")

    _encoder, decoder = build_prediction_vae_models(
        input_shape=input_shape,
        output_shape=output_shape,
        latent_channel=latent_channel,
        channels=channels,
        decoder_downsample_channels=decoder_downsample_channels,
        decoder_context_latent_channel=decoder_context_latent_channel,
        static_stem_channels=static_stem_channels,
        velocity_mlp_dim=velocity_mlp_dim,
        encoder_velocity_condition_channels=encoder_velocity_condition_channels,
        decoder_velocity_condition_channels=decoder_velocity_condition_channels,
        downsample_strides=model_downsample_strides,
        upsample_strides=model_upsample_strides,
        upsample_channels=model_upsample_channels,
        device=device,
    )

    if "decoder" not in ckpt:
        raise ValueError("Checkpoint must contain key: 'decoder'")

    decoder.load_state_dict(ckpt["decoder"], strict=True)

    decoder.eval()

    latent_t, latent_h, latent_w = 1, input_shape[2], input_shape[3]
    for stride in model_downsample_strides:
        latent_h = (latent_h + stride[0] - 1) // stride[0]
        latent_w = (latent_w + stride[1] - 1) // stride[1]

    return decoder, history_len, decoder_context_len, (latent_t, latent_h, latent_w), latent_channel


def autoregressive_predict(
    dynamic_sequence: torch.Tensor,
    static_sequence: torch.Tensor,
    velocity_sequence: torch.Tensor,
    decoder: VAEPredictionDecoder,
    history_len: int,
    decoder_context_len: int,
    latent_shape: tuple[int, int, int],
    latent_channels: int,
    horizon: int,
    num_modalities: int,
    device: torch.device,
    binary_feedback: bool,
    threshold: float,
) -> torch.Tensor:
    """Compute predictions for all anchors t in [history_len, T-1].

    Returns tensor of shape (M, T-history_len, horizon, H, W), where M is
    the number of selected modalities.
    """
    t_total, h, w = dynamic_sequence.shape
    if static_sequence.ndim != 3:
        raise ValueError("static_sequence must have shape (T, H, W)")
    if static_sequence.shape != dynamic_sequence.shape:
        raise ValueError("static_sequence shape must match dynamic_sequence shape")
    if t_total <= history_len:
        return torch.zeros((num_modalities, 0, horizon, h, w), dtype=torch.float32)

    all_preds = torch.zeros((num_modalities, t_total - history_len, horizon, h, w), dtype=torch.float32)

    with torch.no_grad():
        for anchor_t in range(history_len, t_total):
            for modality_idx in range(num_modalities):
                context = dynamic_sequence[anchor_t - history_len : anchor_t].clone()  # (history_len, H, W)
                z = torch.randn((1, latent_channels, latent_shape[0], latent_shape[1], latent_shape[2]), device=device)
                pred_frames: list[torch.Tensor] = []
                for _ in range(horizon):
                    x_decoder_dynamic = context[-decoder_context_len:].unsqueeze(0).unsqueeze(0).to(device)
                    x_static = static_sequence[anchor_t].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
                    current_velocity = velocity_sequence[anchor_t].unsqueeze(0).to(device)
                    logits = decoder(z, x_decoder_dynamic, x_static, current_velocity)
                    prob = torch.sigmoid(logits)[0, 0, 0].detach().cpu()
                    pred_frames.append(prob)

                    feedback = (prob >= threshold).float() if binary_feedback else prob
                    context = torch.cat([context, feedback.unsqueeze(0)], dim=0)

                all_preds[modality_idx, anchor_t - history_len] = torch.stack(pred_frames, dim=0)

    return all_preds


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive visualizer for VAE occupancy predictions")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to rollout .pt file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--scene-index", type=int, default=0, help="Initial scene index")
    parser.add_argument("--agent-index", type=int, default=0, help="Initial agent index within selected scene")
    parser.add_argument("--horizon", type=int, default=4, help="Initial autoregressive horizon")
    parser.add_argument("--max-horizon", type=int, default=16, help="Max horizon in GUI slider")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--binary-feedback", action="store_true", help="Feed thresholded predictions back into context")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary feedback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    decoder, history_len, decoder_context_len, latent_shape, latent_channels = build_models(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    (
        scene_sequences,
        scene_static_maps,
        scene_velocity_sequences,
        scene_anchor_times,
    ) = load_scene_origins(
        args.data_file,
        history_len=history_len,
        future_len=max(1, int(args.max_horizon)),
    )
    if not scene_sequences:
        raise ValueError(f"No scene sequences found in {args.data_file}")

    for scene_idx, agents in enumerate(scene_sequences):
        if not agents:
            raise ValueError(f"Scene {scene_idx} has no agents")
        for agent_idx, seq in enumerate(agents):
            if seq.ndim != 4:
                raise ValueError(
                    f"Scene {scene_idx}, agent {agent_idx} has invalid shape {tuple(seq.shape)}; expected (A,T,H,W)"
                )

    init_scene = clamp_int(args.scene_index, 0, len(scene_sequences) - 1)
    init_agent = clamp_int(args.agent_index, 0, len(scene_sequences[init_scene]) - 1)
    init_seq = scene_sequences[init_scene][init_agent][0]
    _, init_h, init_w = init_seq.shape

    # Cache by (scene_index, agent_index, anchor_index, horizon, num_modalities).
    cache: dict[tuple[int, int, int, int, int], torch.Tensor] = {}

    def get_predictions(
        scene_idx: int,
        agent_idx: int,
        anchor_idx: int,
        horizon: int,
        num_modalities: int,
    ) -> torch.Tensor:
        key = (scene_idx, agent_idx, anchor_idx, horizon, num_modalities)
        if key not in cache:
            print(
                f"Running inference for scene={scene_idx}, agent={agent_idx}, anchor={anchor_idx}, "
                f"horizon={horizon}, modalities={num_modalities} ..."
            )
            seq = scene_sequences[scene_idx][agent_idx][anchor_idx]
            static_sequence = scene_static_maps[scene_idx][agent_idx][anchor_idx]
            velocity_seq = scene_velocity_sequences[scene_idx][agent_idx][anchor_idx]
            cache[key] = autoregressive_predict(
                dynamic_sequence=seq,
                static_sequence=static_sequence,
                velocity_sequence=velocity_seq,
                decoder=decoder,
                history_len=history_len,
                decoder_context_len=decoder_context_len,
                latent_shape=latent_shape,
                latent_channels=latent_channels,
                horizon=horizon,
                num_modalities=num_modalities,
                device=device,
                binary_feedback=args.binary_feedback,
                threshold=args.threshold,
            )
            t_total = int(seq.shape[0])
            print(
                "Done. Total inferences: "
                f"{(max(t_total - history_len, 0)) * horizon * num_modalities}"
            )
        return cache[key]

    # Build figure and axes.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_past, ax_pred, ax_overlay_pred, ax_overlay_gt = axes.flatten()
    plt.subplots_adjust(bottom=0.30)

    zero_rgb = np.zeros((init_h, init_w, 3), dtype=np.float32)
    # Occupancy is cell-discrete; nearest interpolation prevents visual blurring.
    im_past = ax_past.imshow(zero_rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    im_pred = ax_pred.imshow(zero_rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    im_overlay_pred = ax_overlay_pred.imshow(zero_rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    im_overlay_gt = ax_overlay_gt.imshow(zero_rgb, vmin=0.0, vmax=1.0, interpolation="nearest")

    ax_past.set_title("Past + Static Stack")
    ax_pred.set_title("Predicted Horizon Stack (Mode 0=Red, Mode 1=Green)")
    ax_overlay_pred.set_title("Overlay: Static (Gray) + Past (Blue) + Mode0 (Red) + Mode1 (Green)")
    ax_overlay_gt.set_title("Overlay: Past (Blue) + GT Future (Green)")

    for ax in [ax_past, ax_pred, ax_overlay_pred, ax_overlay_gt]:
        ax.set_xticks([])
        ax.set_yticks([])

    # Sliders.
    slider_color = "lightgoldenrodyellow"
    ax_mode = plt.axes([0.02, 0.10, 0.09, 0.14], facecolor=slider_color)
    ax_scene = plt.axes([0.12, 0.12, 0.18, 0.10], facecolor=slider_color)
    ax_origin = plt.axes([0.34, 0.17, 0.56, 0.03], facecolor=slider_color)
    ax_t = plt.axes([0.12, 0.09, 0.78, 0.03], facecolor=slider_color)
    ax_h = plt.axes([0.12, 0.04, 0.78, 0.03], facecolor=slider_color)
    ax_resample = plt.axes([0.02, 0.03, 0.09, 0.05])

    max_mode_count = 2
    mode_labels = [f"Mode {i}" for i in range(max_mode_count)]
    mode_checks = CheckButtons(ax_mode, labels=mode_labels, actives=[True, False])
    ax_mode.set_title("Modes", fontsize=9)
    mode_state = {"selected": {0}}

    scene_options = [str(i) for i in range(len(scene_sequences))]
    scene_radio = RadioButtons(ax_scene, labels=scene_options, active=init_scene)
    ax_scene.set_title("Scene", fontsize=9)
    scene_state = {"index": init_scene}

    agent_slider = Slider(
        ax_origin,
        "Agent ID",
        0,
        max(0, len(scene_sequences[init_scene]) - 1),
        valinit=init_agent,
        valstep=1,
    )

    init_anchor_max = max(0, int(scene_sequences[init_scene][init_agent].shape[0]) - 1)
    t_slider = Slider(ax_t, "Anchor", 0, init_anchor_max, valinit=0, valstep=1)

    h_slider = Slider(
        ax_h,
        "Horizon",
        1,
        max(1, args.max_horizon),
        valinit=clamp_int(args.horizon, 1, max(1, args.max_horizon)),
        valstep=1,
    )
    resample_button = Button(ax_resample, "Resample z")

    info_text = fig.text(0.02, 0.96, "", fontsize=10, va="top")

    def _on_scene_select(selection: str) -> None:
        try:
            scene_state["index"] = int(selection)
        except ValueError:
            return
        refresh(None)

    def _on_mode_select(selection: str) -> None:
        if selection not in mode_labels:
            return
        mode_idx = mode_labels.index(selection)
        selected_modes = mode_state["selected"]
        if mode_idx in selected_modes:
            if len(selected_modes) == 1:
                # Keep at least one mode visible.
                mode_checks.set_active(mode_idx)
                return
            selected_modes.remove(mode_idx)
        else:
            selected_modes.add(mode_idx)
        refresh(None)

    def refresh(_: float | None = None) -> None:
        scene_idx = int(scene_state["index"])
        agent_max = max(0, len(scene_sequences[scene_idx]) - 1)
        if agent_slider.valmax != agent_max:
            agent_slider.valmax = agent_max
            agent_slider.ax.set_xlim(agent_slider.valmin, agent_max)

        agent_req = int(agent_slider.val)
        agent_idx = clamp_int(agent_req, 0, agent_max)
        if agent_idx != agent_req:
            agent_slider.set_val(agent_idx)
            return

        horizon = int(h_slider.val)
        selected_modes = sorted(mode_state["selected"])

        agent_anchor_count = int(scene_sequences[scene_idx][agent_idx].shape[0])
        anchor_max = max(0, agent_anchor_count - 1)
        if t_slider.valmax != anchor_max:
            t_slider.valmax = anchor_max
            t_slider.ax.set_xlim(t_slider.valmin, anchor_max)

        anchor_req = int(t_slider.val)
        anchor_idx = clamp_int(anchor_req, 0, anchor_max)
        if anchor_idx != anchor_req:
            t_slider.set_val(anchor_idx)
            return

        seq = scene_sequences[scene_idx][agent_idx][anchor_idx]
        static_seq = scene_static_maps[scene_idx][agent_idx][anchor_idx]
        t_total, h_img, w_img = seq.shape

        preds = get_predictions(scene_idx, agent_idx, anchor_idx, horizon, max_mode_count)

        t_anchor = int(history_len)
        past = seq[t_anchor - history_len : t_anchor]  # (history_len, H, W)
        past_stack = (past.max(dim=0).values > 0.0).float()
        static_map = (static_seq[t_anchor] > 0.0).float()

        pred_stack_1 = torch.zeros((h_img, w_img), dtype=torch.float32)
        pred_stack_2 = torch.zeros((h_img, w_img), dtype=torch.float32)
        if preds.shape[1] > 0:
            pred_index = 0
            if 0 in selected_modes and preds.shape[0] > 0:
                pred_seq_1 = preds[0, pred_index]  # (horizon, H, W)
                pred_stack_1 = pred_seq_1.max(dim=0).values
            if 1 in selected_modes and preds.shape[0] > 1:
                pred_seq_2 = preds[1, pred_index]
                pred_stack_2 = pred_seq_2.max(dim=0).values

        gt_future = seq[t_anchor : min(t_anchor + horizon, t_total)]
        if gt_future.shape[0] > 0:
            gt_stack = (gt_future.max(dim=0).values > 0.0).float()
        else:
            gt_stack = torch.zeros((h_img, w_img), dtype=torch.float32)

        past_np = past_stack.numpy()
        static_np = static_map.numpy()
        pred1_np = pred_stack_1.numpy()
        pred2_np = pred_stack_2.numpy()
        gt_np = gt_stack.numpy()

        # Show static occupancy as gray background and past occupancy in blue.
        past_with_static = np.stack(
            [0.35 * static_np, 0.35 * static_np, np.clip(0.35 * static_np + past_np, 0.0, 1.0)],
            axis=-1,
        )

        pred_viz = np.stack([pred1_np, pred2_np, np.zeros_like(pred1_np)], axis=-1)
        overlay_pred = np.stack(
            [
                np.clip(0.35 * static_np + pred1_np, 0.0, 1.0),
                np.clip(0.35 * static_np + pred2_np, 0.0, 1.0),
                np.clip(0.35 * static_np + past_np, 0.0, 1.0),
            ],
            axis=-1,
        )
        overlay_gt = np.stack([np.zeros_like(gt_np), gt_np, past_np], axis=-1)

        im_past.set_data(np.clip(past_with_static, 0.0, 1.0))
        im_pred.set_data(np.clip(pred_viz, 0.0, 1.0))
        im_overlay_pred.set_data(np.clip(overlay_pred, 0.0, 1.0))
        im_overlay_gt.set_data(np.clip(overlay_gt, 0.0, 1.0))

        ax_past.set_title(f"Past+Static Stack (anchor={anchor_idx}, hist={history_len})")
        selected_label = ",".join(str(m) for m in selected_modes)
        ax_pred.set_title(f"Pred Stack (h={horizon}, modes=[{selected_label}])")

        anchor_time = scene_anchor_times[scene_idx][agent_idx][anchor_idx]

        total_infer = max(t_total - history_len, 0) * horizon * max_mode_count
        info_text.set_text(
            f"scene={scene_idx} | agent={agent_idx} | anchor_idx={anchor_idx} | anchor_t={anchor_time} | "
            f"window_T={t_total} | horizon={horizon} | "
            f"selected_modes=[{selected_label}] | total inferences={total_infer}"
        )

        fig.canvas.draw_idle()

    def _on_resample_click(_: object) -> None:
        cache.clear()
        print("Cleared prediction cache; recomputing with freshly sampled latent z.")
        refresh(None)

    mode_checks.on_clicked(_on_mode_select)
    scene_radio.on_clicked(_on_scene_select)
    agent_slider.on_changed(refresh)
    t_slider.on_changed(refresh)
    h_slider.on_changed(refresh)
    resample_button.on_clicked(_on_resample_click)

    refresh(None)
    plt.show()


if __name__ == "__main__":
    main()
