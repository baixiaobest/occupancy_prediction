from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualize_model import build_models, load_scene_origins
from src.rollout_data import RollOutData
from src.rl.counterfactual import rollout_counterfactual_futures, sample_random_velocity_plans


def _overlay_rgb(*, static_map: torch.Tensor, past_map: torch.Tensor, future_map: torch.Tensor) -> np.ndarray:
    static_np = torch.as_tensor(static_map, dtype=torch.float32).detach().cpu().numpy()
    past_np = torch.as_tensor(past_map, dtype=torch.float32).detach().cpu().numpy()
    future_np = torch.as_tensor(future_map, dtype=torch.float32).detach().cpu().numpy()

    static_np = np.clip(static_np, 0.0, 1.0)
    past_np = np.clip(past_np, 0.0, 1.0)
    future_np = np.clip(future_np, 0.0, 1.0)

    rgb = np.zeros((*static_np.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.maximum(static_np * 0.65, future_np)
    rgb[..., 1] = np.maximum(static_np * 0.65, 0.0)
    rgb[..., 2] = np.maximum(static_np * 0.65, past_np)
    return np.clip(rgb, 0.0, 1.0)


def _load_scene_resolutions(data_file: Path) -> list[tuple[float, float]]:
    payload = torch.load(data_file, map_location="cpu")
    if not isinstance(payload, RollOutData):
        raise ValueError(f"Unsupported payload format in {data_file}: expected RollOutData")

    resolutions: list[tuple[float, float]] = []
    for scene in payload.scenes:
        dynamic_maps = getattr(scene, "scene_dynamic_maps", None)
        if dynamic_maps is None:
            continue
        dynamic_tensor = torch.as_tensor(dynamic_maps)
        if dynamic_tensor.ndim != 4 or int(dynamic_tensor.shape[0]) == 0:
            continue
        resolutions.append((float(scene.occupancy_resolution[0]), float(scene.occupancy_resolution[1])))
    return resolutions


def _plot_trajectory_panel(
    ax: plt.Axes,
    *,
    position_offsets: torch.Tensor,
    candidate_position_offsets: torch.Tensor,
    horizon: int,
    time_index: int,
    static_map: torch.Tensor,
    past_map: torch.Tensor,
    future_map: torch.Tensor,
    resolution_xy: tuple[float, float],
) -> None:
    history_len = int(position_offsets.shape[0] - horizon)
    past_offsets = position_offsets[: history_len + 1].detach().cpu().numpy()
    gt_future_offsets = position_offsets[history_len : history_len + horizon].detach().cpu().numpy()
    candidate_offsets = candidate_position_offsets.detach().cpu().numpy()
    time_index = max(0, min(int(time_index), horizon - 1))

    gt_traj = gt_future_offsets[: time_index + 1]

    overlay_rgb = _overlay_rgb(static_map=static_map, past_map=past_map, future_map=future_map)
    cells_y, cells_x = overlay_rgb.shape[:2]
    res_x, res_y = float(resolution_xy[0]), float(resolution_xy[1])
    extent = (
        -0.5 * cells_x * res_x,
        0.5 * cells_x * res_x,
        -0.5 * cells_y * res_y,
        0.5 * cells_y * res_y,
    )
    ax.imshow(
        overlay_rgb,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        alpha=0.75,
        zorder=0,
    )

    ax.plot(past_offsets[:, 0], past_offsets[:, 1], color="tab:blue", linewidth=2.0, label="past", zorder=3)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], color="tab:green", linewidth=2.0, label="gt future", zorder=3)
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color="tab:green", s=36, zorder=4)
    ax.scatter([0.0], [0.0], color="black", s=50, marker="o", label="anchor", zorder=4)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, candidate_offsets.shape[0])))
    for candidate_idx, (traj, color) in enumerate(zip(candidate_offsets, colors)):
        traj_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), traj[: time_index + 1]], axis=0)
        ax.plot(
            traj_xy[:, 0],
            traj_xy[:, 1],
            color=color,
            linewidth=1.75,
            alpha=0.9,
            label=f"candidate {candidate_idx}",
            zorder=3,
        )
        ax.scatter(traj_xy[-1, 0], traj_xy[-1, 1], color=color, s=28, zorder=4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Trajectory Candidates (ego frame) | step {time_index + 1}/{horizon}")
    ax.set_xlabel("x offset [m]")
    ax.set_ylabel("y offset [m]")
    ax.legend(loc="best", fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random counterfactual trajectories and predicted occupancy")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to rollout .pt file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to VAE checkpoint .pt")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index in rollout file")
    parser.add_argument("--agent-index", type=int, default=0, help="Agent index within the selected scene")
    parser.add_argument("--anchor-index", type=int, default=0, help="Anchor/window index for selected agent")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction horizon to visualize")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of random candidate trajectories")
    parser.add_argument("--max-speed", type=float, default=2.0, help="Clamp random candidate speeds to this value")
    parser.add_argument("--delta-std", type=float, default=0.35, help="Random-walk velocity noise std")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--binary-feedback", action="store_true", help="Feed thresholded predictions back into context")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary feedback")
    parser.add_argument("--tap-layer", type=int, default=None, help="Optional decoder tap layer to compute")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for candidate trajectories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0")
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be > 0")
    if args.max_speed <= 0.0:
        raise ValueError("--max-speed must be > 0")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    decoder, history_len, decoder_context_len, latent_shape, latent_channels = build_models(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    (
        scene_sequences,
        scene_static_maps,
        scene_velocity_sequences,
        scene_position_offsets,
        _scene_anchor_times,
    ) = load_scene_origins(
        args.data_file,
        history_len=history_len,
        future_len=max(1, args.horizon),
    )
    scene_resolutions = _load_scene_resolutions(args.data_file)

    if not scene_sequences:
        raise ValueError(f"No scene sequences found in {args.data_file}")

    state = {
        "scene_index": max(0, min(int(args.scene_index), len(scene_sequences) - 1)),
        "agent_index": 0,
        "anchor_index": 0,
        "bundle": None,
    }

    def _load_bundle(scene_index: int, anchor_index: int) -> dict[str, torch.Tensor | int]:
        scene_index = max(0, min(int(scene_index), len(scene_sequences) - 1))
        agent_index = max(0, min(int(args.agent_index), len(scene_sequences[scene_index]) - 1))
        anchor_count = int(scene_sequences[scene_index][agent_index].shape[0])
        anchor_index = max(0, min(int(anchor_index), anchor_count - 1))

        dynamic_seq = scene_sequences[scene_index][agent_index][anchor_index].to(dtype=torch.float32)
        static_seq = scene_static_maps[scene_index][agent_index][anchor_index].to(dtype=torch.float32)
        velocity_seq = scene_velocity_sequences[scene_index][agent_index][anchor_index].to(dtype=torch.float32)
        position_offsets = scene_position_offsets[scene_index][agent_index][anchor_index].to(dtype=torch.float32)

        max_horizon = int(dynamic_seq.shape[0] - history_len)
        horizon = max(1, min(int(args.horizon), max_horizon))

        dynamic_context = dynamic_seq[history_len - decoder_context_len : history_len].unsqueeze(0).unsqueeze(0).to(device)
        static_map = static_seq[history_len].unsqueeze(0).unsqueeze(0).to(device)
        current_velocity = velocity_seq[history_len].unsqueeze(0).to(device)
        current_position_offset = position_offsets[history_len].unsqueeze(0).to(device)

        local_generator = torch.Generator(device=device)
        local_generator.manual_seed(
            int(args.seed) + 1009 * scene_index + 7919 * agent_index + 104729 * anchor_index
        )
        candidate_plans = sample_random_velocity_plans(
            current_velocity=current_velocity,
            num_candidates=int(args.num_candidates),
            horizon=horizon,
            max_speed=float(args.max_speed),
            delta_std=float(args.delta_std),
            dt=0.1,
            generator=local_generator,
        )

        rollout = rollout_counterfactual_futures(
            decoder=decoder,
            dynamic_context=dynamic_context,
            static_map=static_map,
            candidate_velocity_plans=candidate_plans,
            latent_channels=latent_channels,
            latent_shape=latent_shape,
            dt=0.1,
            current_position_offset=current_position_offset,
            tap_layer=args.tap_layer,
            binary_feedback=bool(args.binary_feedback),
            threshold=float(args.threshold),
        )

        predicted_probs = torch.sigmoid(rollout.predicted_logits[0])
        return {
            "scene_index": scene_index,
            "agent_index": agent_index,
            "anchor_index": anchor_index,
            "anchor_count": anchor_count,
            "horizon": horizon,
            "resolution_xy": scene_resolutions[scene_index],
            "dynamic_seq": dynamic_seq,
            "position_offsets": position_offsets,
            "candidate_offsets": rollout.candidate_position_offsets[0],
            "candidate_plans": candidate_plans[0].detach().cpu(),
            "predicted_future_steps": predicted_probs[:, :, 0].detach().cpu(),
            "past_stack": dynamic_seq[:history_len].amax(dim=0),
            "gt_future_steps": dynamic_seq[history_len : history_len + horizon],
            "static_anchor": static_seq[history_len],
        }

    fig_traj, trajectory_ax = plt.subplots(figsize=(9.5, 9.0))
    fig_traj.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.28)

    occupancy_panel_count = int(args.num_candidates) + 1
    occ_cols = min(3, occupancy_panel_count)
    occ_rows = int(math.ceil(occupancy_panel_count / occ_cols))
    fig_occ, occ_axes = plt.subplots(
        occ_rows,
        occ_cols,
        figsize=(6.8 * occ_cols, 5.8 * occ_rows),
        squeeze=False,
    )
    occ_axes_flat = occ_axes.flatten()
    fig_occ.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.07, wspace=0.28, hspace=0.34)

    gt_ax = occ_axes_flat[0]

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, int(args.num_candidates))))
    candidate_axes: list[plt.Axes] = []
    for candidate_idx in range(int(args.num_candidates)):
        ax = occ_axes_flat[candidate_idx + 1]
        candidate_axes.append(ax)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[candidate_idx])
            spine.set_linewidth(2.0)

    for ax in occ_axes_flat[occupancy_panel_count:]:
        ax.axis("off")

    suppress_callbacks = {"active": False}

    scene_prev_ax = fig_traj.add_axes([0.10, 0.16, 0.12, 0.05])
    scene_next_ax = fig_traj.add_axes([0.25, 0.16, 0.12, 0.05])
    anchor_slider_ax = fig_traj.add_axes([0.46, 0.17, 0.44, 0.035])
    slider_ax = fig_traj.add_axes([0.18, 0.09, 0.72, 0.035])

    scene_prev_button = Button(scene_prev_ax, "Prev scene")
    scene_next_button = Button(scene_next_ax, "Next scene")
    scene_label = fig_traj.text(0.10, 0.235, "", fontsize=10)

    initial_bundle = _load_bundle(state["scene_index"], int(args.anchor_index))
    state["scene_index"] = int(initial_bundle["scene_index"])
    state["agent_index"] = int(initial_bundle["agent_index"])
    state["anchor_index"] = int(initial_bundle["anchor_index"])
    state["bundle"] = initial_bundle

    anchor_slider = Slider(
        anchor_slider_ax,
        "Rollout time",
        0,
        max(0, int(initial_bundle["anchor_count"]) - 1),
        valinit=int(initial_bundle["anchor_index"]),
        valstep=1,
    )

    time_slider = Slider(
        slider_ax,
        "Future step",
        0,
        int(initial_bundle["horizon"]) - 1,
        valinit=0,
        valstep=1,
    )

    def _update_slider_range(slider: Slider, new_max: int) -> None:
        slider.valmax = float(new_max)
        slider.ax.set_xlim(slider.valmin, float(new_max))

    def _refresh_bundle() -> None:
        bundle = _load_bundle(state["scene_index"], state["anchor_index"])
        state["scene_index"] = int(bundle["scene_index"])
        state["agent_index"] = int(bundle["agent_index"])
        state["anchor_index"] = int(bundle["anchor_index"])
        state["bundle"] = bundle

        suppress_callbacks["active"] = True
        _update_slider_range(anchor_slider, max(0, int(bundle["anchor_count"]) - 1))
        if int(anchor_slider.val) != int(bundle["anchor_index"]):
            anchor_slider.set_val(int(bundle["anchor_index"]))
        _update_slider_range(time_slider, int(bundle["horizon"]) - 1)
        clamped_time = max(0, min(int(time_slider.val), int(bundle["horizon"]) - 1))
        if int(time_slider.val) != clamped_time:
            time_slider.set_val(clamped_time)
        suppress_callbacks["active"] = False

    def _render_step(step_index: int) -> None:
        bundle = state["bundle"]
        if bundle is None:
            raise RuntimeError("Visualization bundle is not initialized")

        horizon = int(bundle["horizon"])
        step_index = max(0, min(int(step_index), horizon - 1))

        trajectory_ax.clear()
        _plot_trajectory_panel(
            trajectory_ax,
            position_offsets=bundle["position_offsets"][: history_len + horizon],
            candidate_position_offsets=bundle["candidate_offsets"],
            horizon=horizon,
            time_index=step_index,
            static_map=bundle["static_anchor"],
            past_map=bundle["past_stack"],
            future_map=bundle["gt_future_steps"][step_index],
            resolution_xy=bundle["resolution_xy"],
        )

        gt_ax.clear()
        gt_rgb = _overlay_rgb(
            static_map=bundle["static_anchor"],
            past_map=bundle["past_stack"],
            future_map=bundle["gt_future_steps"][step_index],
        )
        gt_ax.imshow(gt_rgb, origin="lower", interpolation="nearest")
        gt_ax.set_title(f"Ground Truth Future Occupancy | step {step_index + 1}/{horizon}")
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])

        for candidate_idx, ax in enumerate(candidate_axes):
            ax.clear()
            pred_rgb = _overlay_rgb(
                static_map=bundle["static_anchor"],
                past_map=bundle["past_stack"],
                future_map=bundle["predicted_future_steps"][candidate_idx, step_index],
            )
            ax.imshow(pred_rgb, origin="lower", interpolation="nearest")
            offset_xy = bundle["candidate_offsets"][candidate_idx, step_index].detach().cpu().numpy()
            speed = torch.linalg.vector_norm(bundle["candidate_plans"][candidate_idx, step_index], dim=-1).item()
            ax.set_title(
                f"Candidate {candidate_idx} | step {step_index + 1}/{horizon}\n"
                f"pos=({offset_xy[0]:.2f}, {offset_xy[1]:.2f}) m, |v|={speed:.2f}"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[candidate_idx])
                spine.set_linewidth(2.0)

        scene_label.set_text(
            f"Scene {int(bundle['scene_index'])} | Agent {int(bundle['agent_index'])} | Rollout time {int(bundle['anchor_index'])}/{int(bundle['anchor_count']) - 1}"
        )
        fig_traj.suptitle(
            f"Random Counterfactuals Trajectories | scene={int(bundle['scene_index'])}, agent={int(bundle['agent_index'])}, anchor={int(bundle['anchor_index'])}, horizon={horizon}",
            fontsize=14,
        )
        fig_occ.suptitle(
            f"Random Counterfactuals | scene={int(bundle['scene_index'])}, agent={int(bundle['agent_index'])}, anchor={int(bundle['anchor_index'])}, horizon={horizon}",
            fontsize=14,
        )
        fig_traj.canvas.draw_idle()
        fig_occ.canvas.draw_idle()

    def _on_slider_change(value: float) -> None:
        if suppress_callbacks["active"]:
            return
        _render_step(int(value))

    def _on_anchor_change(value: float) -> None:
        if suppress_callbacks["active"]:
            return
        state["anchor_index"] = int(value)
        _refresh_bundle()
        _render_step(int(time_slider.val))

    def _change_scene(delta: int) -> None:
        state["scene_index"] = max(0, min(state["scene_index"] + int(delta), len(scene_sequences) - 1))
        state["anchor_index"] = 0
        _refresh_bundle()
        _render_step(0)

    time_slider.on_changed(_on_slider_change)
    anchor_slider.on_changed(_on_anchor_change)
    scene_prev_button.on_clicked(lambda _event: _change_scene(-1))
    scene_next_button.on_clicked(lambda _event: _change_scene(1))
    _render_step(0)

    plt.show()


if __name__ == "__main__":
    main()
