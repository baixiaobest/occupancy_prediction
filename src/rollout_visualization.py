from __future__ import annotations

import importlib
from typing import List, Tuple

import numpy as np
import torch

from src.scene import ObstacleSpec, PathSpec


def prepare_animation_grids(
    static_maps: List[List[torch.Tensor]],
) -> List[List[np.ndarray]]:
    return [
        [grid.detach().cpu().numpy() for grid in per_center_static]
        for per_center_static in static_maps
    ]


def prepare_past_future_dynamic_grids(
    dynamic_windows: List[List[List[torch.Tensor]]],
    frame_offsets: List[int],
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    past_grids: List[List[np.ndarray]] = []
    future_grids: List[List[np.ndarray]] = []

    for agent_windows in dynamic_windows:
        agent_past: List[np.ndarray] = []
        agent_future: List[np.ndarray] = []
        for time_frames in agent_windows:
            if not time_frames:
                continue
            if len(time_frames) != len(frame_offsets):
                raise ValueError("time frame count must match frame_offsets length")

            frame_tensors = [torch.as_tensor(frame, dtype=torch.float32) for frame in time_frames]
            zero_grid = torch.zeros_like(frame_tensors[0])
            past_frames = [frame_tensors[idx] for idx, dt in enumerate(frame_offsets) if dt < 0]
            future_frames = [frame_tensors[idx] for idx, dt in enumerate(frame_offsets) if dt >= 0]

            past_stack = torch.stack(past_frames, dim=0).amax(dim=0) if past_frames else zero_grid
            future_stack = torch.stack(future_frames, dim=0).amax(dim=0) if future_frames else zero_grid

            agent_past.append(past_stack.detach().cpu().numpy())
            agent_future.append(future_stack.detach().cpu().numpy())

        past_grids.append(agent_past)
        future_grids.append(agent_future)

    return past_grids, future_grids


def animate_rollout(
    traj: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    paths: List[PathSpec],
    static_maps: List[List[np.ndarray]],
    dynamic_past_grids: List[List[np.ndarray]],
    dynamic_future_grids: List[List[np.ndarray]],
    occupancy_origins: List[np.ndarray],
    occupancy_resolution: Tuple[float, float],
    time_step: float,
    title_prefix: str = "",
) -> None:
    """Animate trajectory and occupancy maps (static + past/future dynamic overlay)."""
    plt = importlib.import_module("matplotlib.pyplot")
    animation_mod = importlib.import_module("matplotlib.animation")
    patches_mod = importlib.import_module("matplotlib.patches")
    FuncAnimation = animation_mod.FuncAnimation
    Polygon = patches_mod.Polygon

    if traj.shape != velocities.shape:
        raise ValueError("traj and velocities must have the same shape")

    num_steps, _, _ = traj.shape

    fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
    traj_title = "ORCA Pedestrian Simulation"
    occ_title = "Static + Dynamic Past/Future Occupancy"
    if title_prefix:
        traj_title = f"{title_prefix} - {traj_title}"
        occ_title = f"{title_prefix} - {occ_title}"

    ax_traj.set_title(traj_title)
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_aspect("equal", adjustable="box")

    min_xy = traj.min(axis=(0, 1))
    max_xy = traj.max(axis=(0, 1))
    min_xy = np.minimum(min_xy, goals.min(axis=0))
    max_xy = np.maximum(max_xy, goals.max(axis=0))
    for obstacle in obstacles:
        if not obstacle.vertices:
            continue
        verts = np.asarray(obstacle.vertices, dtype=np.float32)
        min_xy = np.minimum(min_xy, verts.min(axis=0))
        max_xy = np.maximum(max_xy, verts.max(axis=0))

    min_xy -= 1.0
    max_xy += 1.0
    ax_traj.set_xlim(float(min_xy[0]), float(max_xy[0]))
    ax_traj.set_ylim(float(min_xy[1]), float(max_xy[1]))

    for obstacle in obstacles:
        if len(obstacle.vertices) < 3:
            continue
        patch = Polygon(
            obstacle.vertices,
            closed=True,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.0,
        )
        ax_traj.add_patch(patch)

    for path in paths:
        if len(path.points) < 2:
            continue
        path_points = np.asarray(path.points, dtype=np.float32)
        ax_traj.plot(
            path_points[:, 0],
            path_points[:, 1],
            linestyle="--",
            linewidth=2.0,
            color="tab:orange",
            alpha=0.9,
        )

    scat = ax_traj.scatter(traj[0, :, 0], traj[0, :, 1], s=60, c="tab:blue")
    vel_quiver = ax_traj.quiver(
        traj[0, :, 0],
        traj[0, :, 1],
        velocities[0, :, 0],
        velocities[0, :, 1],
        color="tab:green",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        alpha=0.9,
    )
    ax_traj.scatter(goals[:, 0], goals[:, 1], s=80, c="tab:red", marker="x")
    traj_time_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, va="top")

    num_occ_maps = len(static_maps)
    if num_occ_maps == 0:
        raise ValueError("static_maps must contain at least one occupancy map")
    if len(dynamic_past_grids) != num_occ_maps or len(dynamic_future_grids) != num_occ_maps:
        raise ValueError("dynamic_past_grids and dynamic_future_grids must match static_maps length")

    occ_cols = int(np.ceil(np.sqrt(num_occ_maps)))
    occ_rows = int(np.ceil(num_occ_maps / occ_cols))
    fig_occ, axes_occ = plt.subplots(
        occ_rows,
        occ_cols * 2,
        figsize=(7.6 * occ_cols, 4.0 * occ_rows),
        squeeze=False,
    )
    fig_occ.suptitle(occ_title)

    static_images = []
    static_time_texts = []
    overlay_images = []
    overlay_time_texts = []
    res_x, res_y = occupancy_resolution

    for occ_idx in range(num_occ_maps):
        row = occ_idx // occ_cols
        col = occ_idx % occ_cols
        ax_static = axes_occ[row, 2 * col]
        ax_overlay = axes_occ[row, 2 * col + 1]

        first_static = static_maps[occ_idx][0]
        first_past = dynamic_past_grids[occ_idx][0]
        first_future = dynamic_future_grids[occ_idx][0]

        cells_y, cells_x = first_static.shape
        origin = occupancy_origins[occ_idx]
        extent = (
            float(origin[0]),
            float(origin[0] + cells_x * res_x),
            float(origin[1]),
            float(origin[1] + cells_y * res_y),
        )

        im_static = ax_static.imshow(
            first_static,
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=extent,
            interpolation="nearest",
        )

        first_overlay = np.stack(
            [
                first_past,
                first_future,
                np.zeros_like(first_past),
            ],
            axis=-1,
        )
        im_overlay = ax_overlay.imshow(
            np.clip(first_overlay, 0.0, 1.0),
            origin="lower",
            vmin=0,
            vmax=1,
            extent=extent,
            interpolation="nearest",
        )

        ax_static.set_title(f"Static #{occ_idx}")
        ax_static.set_xlabel("X")
        ax_static.set_ylabel("Y")
        ax_static.set_aspect("equal", adjustable="box")
        time_text = ax_static.text(0.02, 0.98, "", transform=ax_static.transAxes, va="top")
        static_images.append(im_static)
        static_time_texts.append(time_text)

        ax_overlay.set_title(f"Past/Future #{occ_idx}")
        ax_overlay.set_xlabel("X")
        ax_overlay.set_ylabel("Y")
        ax_overlay.set_aspect("equal", adjustable="box")
        overlay_time = ax_overlay.text(0.02, 0.98, "", transform=ax_overlay.transAxes, va="top")
        overlay_images.append(im_overlay)
        overlay_time_texts.append(overlay_time)

    for row in range(occ_rows):
        for col in range(occ_cols):
            occ_idx = row * occ_cols + col
            if occ_idx >= num_occ_maps:
                axes_occ[row, 2 * col].axis("off")
                axes_occ[row, 2 * col + 1].axis("off")

    def update(frame: int):
        sim_step = int(frame)

        scat.set_offsets(traj[sim_step])
        vel_quiver.set_offsets(traj[sim_step])
        vel_quiver.set_UVC(velocities[sim_step, :, 0], velocities[sim_step, :, 1])
        traj_time_text.set_text(f"t={sim_step * time_step:.2f}s")
        artists = [scat, vel_quiver, traj_time_text]
        for occ_idx, (im_static, time_text_static) in enumerate(zip(static_images, static_time_texts)):
            im_static.set_data(static_maps[occ_idx][sim_step])
            time_text_static.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={sim_step * time_step:.2f}s"
            )
            artists.extend([im_static, time_text_static])
        for occ_idx, (im_overlay, time_text_overlay) in enumerate(zip(overlay_images, overlay_time_texts)):
            overlay_rgb = np.stack(
                [
                    dynamic_past_grids[occ_idx][sim_step],
                    dynamic_future_grids[occ_idx][sim_step],
                    np.zeros_like(dynamic_past_grids[occ_idx][sim_step]),
                ],
                axis=-1,
            )
            im_overlay.set_data(np.clip(overlay_rgb, 0.0, 1.0))
            time_text_overlay.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={sim_step * time_step:.2f}s"
            )
            artists.extend([im_overlay, time_text_overlay])
        return tuple(artists)

    anim_traj = FuncAnimation(
        fig_traj,
        update,
        frames=num_steps,
        interval=time_step * 1000,
        blit=False,
    )
    anim_occ = FuncAnimation(
        fig_occ,
        update,
        frames=num_steps,
        interval=time_step * 1000,
        blit=False,
    )

    plt.tight_layout()
    plt.show()

    _ = (anim_traj, anim_occ)
