from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import os
import sys
import torch

# Ensure project root is on sys.path so `from src...` works when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scene import ObstacleSpec, PathSpec
from src.ORCASim import ORCASim
from src.occupancy2d import Occupancy2d
from src.rollout_data import AgentRollOutData, RollOutData, SceneRollOutData
from src.rollout_setting import RollOutSetting
from src.templates import default_templates, test_templates


# RollOutSetting is defined in src.rollout_setting


def _compute_scene_canvas(
    traj: np.ndarray,
    obstacles: List[ObstacleSpec],
    resolution: float,
    occupancy_length: float,
    occupancy_width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], np.ndarray]:
    """Compute global scene canvas origin/size/grid shape/center.

    The canvas is expanded by half of the local crop size so any anchor-centered
    local crop can be safely sliced from the stored global maps.
    """
    obs_min_x = np.inf
    obs_max_x = -np.inf
    obs_min_y = np.inf
    obs_max_y = -np.inf
    for obstacle in obstacles:
        if not obstacle.vertices:
            continue
        verts = np.asarray(obstacle.vertices, dtype=np.float32)
        obs_min_x = min(obs_min_x, float(np.min(verts[:, 0])))
        obs_max_x = max(obs_max_x, float(np.max(verts[:, 0])))
        obs_min_y = min(obs_min_y, float(np.min(verts[:, 1])))
        obs_max_y = max(obs_max_y, float(np.max(verts[:, 1])))

    traj_min_x = float(np.min(traj[:, :, 0]))
    traj_max_x = float(np.max(traj[:, :, 0]))
    traj_min_y = float(np.min(traj[:, :, 1]))
    traj_max_y = float(np.max(traj[:, :, 1]))

    if np.isfinite(obs_min_x):
        min_x_base = min(traj_min_x, obs_min_x)
        max_x_base = max(traj_max_x, obs_max_x)
        min_y_base = min(traj_min_y, obs_min_y)
        max_y_base = max(traj_max_y, obs_max_y)
    else:
        min_x_base = traj_min_x
        max_x_base = traj_max_x
        min_y_base = traj_min_y
        max_y_base = traj_max_y

    half_local_w_m = 0.5 * float(occupancy_length)
    half_local_h_m = 0.5 * float(occupancy_width)
    scene_min_x = min_x_base - half_local_w_m
    scene_max_x = max_x_base + half_local_w_m
    scene_min_y = min_y_base - half_local_h_m
    scene_max_y = max_y_base + half_local_h_m

    scene_cells_x = max(1, int(np.ceil((scene_max_x - scene_min_x) / float(resolution))))
    scene_cells_y = max(1, int(np.ceil((scene_max_y - scene_min_y) / float(resolution))))
    scene_size = (float(scene_cells_x * resolution), float(scene_cells_y * resolution))
    scene_origin = (float(scene_min_x), float(scene_min_y))
    scene_center = np.array(
        [scene_origin[0] + 0.5 * scene_size[0], scene_origin[1] + 0.5 * scene_size[1]],
        dtype=np.float32,
    )
    return scene_origin, scene_size, (scene_cells_y, scene_cells_x), scene_center


def _build_scene_static_map(
    obstacles: List[ObstacleSpec],
    resolution: float,
    agent_radius: float,
    scene_size: Tuple[float, float],
    scene_center: np.ndarray,
) -> torch.Tensor:
    """Rasterize one static occupancy map for the full scene canvas."""
    static_occ = Occupancy2d(
        resolution=(resolution, resolution),
        size=scene_size,
        trajectory=None,
        static_obstacles=obstacles,
        agent_radius=agent_radius,
    )
    return static_occ.generate(center_offset=scene_center)[0].to(dtype=torch.float32)


def _build_agent_dynamic_map(
    traj: np.ndarray,
    center_agent_idx: int,
    resolution: float,
    agent_radius: float,
    scene_size: Tuple[float, float],
    scene_center: np.ndarray,
    canvas_shape: Tuple[int, int],
) -> torch.Tensor:
    """Rasterize global dynamic occupancy over all timesteps for one centered agent.

    The centered agent is excluded and all other agents are rasterized.
    """
    num_steps = int(traj.shape[0])
    num_agents = int(traj.shape[1])
    other_agent_indices = [idx for idx in range(num_agents) if idx != center_agent_idx]

    if not other_agent_indices:
        return torch.zeros((num_steps, canvas_shape[0], canvas_shape[1]), dtype=torch.float32)

    other_traj = traj[:, other_agent_indices].astype(np.float32)
    dynamic_occ = Occupancy2d(
        resolution=(resolution, resolution),
        size=scene_size,
        trajectory=other_traj,
        static_obstacles=None,
        agent_radius=agent_radius,
    )
    generated_frames = dynamic_occ.generate(center_offset=scene_center)
    return torch.stack([grid.to(dtype=torch.float32) for grid in generated_frames], dim=0)


def _collect_anchor_metadata(
    traj: np.ndarray,
    velocities: np.ndarray,
    center_agent_idx: int,
    anchor_steps: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect anchor center positions and anchor velocities for one agent."""
    velocity_series = torch.zeros((len(anchor_steps), 2), dtype=torch.float32)
    center_series = torch.zeros((len(anchor_steps), 2), dtype=torch.float32)

    for anchor_idx, anchor_t in enumerate(anchor_steps):
        center_series[anchor_idx] = torch.as_tensor(
            traj[anchor_t, center_agent_idx], dtype=torch.float32
        )
        velocity_series[anchor_idx] = torch.as_tensor(
            velocities[anchor_t, center_agent_idx], dtype=torch.float32
        )

    return center_series, velocity_series

def build_agent_centric_occupancy_sequences(
    traj: np.ndarray,
    velocities: np.ndarray,
    obstacles: List[ObstacleSpec],
    resolution: float,
    agent_radius: float,
    occupancy_length: float,
    occupancy_width: float,
    sample_interval: int,
    past_frames: int,
    future_frames: int,
    center_agent_indices: List[int] | None = None,
) -> Tuple[
    List[torch.Tensor],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[float, float],
    Tuple[float, float],
    List[int],
    List[int],
    List[torch.Tensor],
    Tuple[int, int],
]:
    """Build scene-global occupancy maps and per-agent anchor metadata.

    Output structure:
    - dynamic_maps[agent] -> (T, H, W) global dynamic occupancy over full time.
    - scene_static_map -> (H, W) global static map for the whole scene.
    - current_velocities[agent][anchor] and anchor_centers[agent][anchor] keep
      sampled anchor metadata for training.
    """
    if occupancy_length <= 0 or occupancy_width <= 0:
        raise ValueError("occupancy_length and occupancy_width must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be > 0")
    if past_frames < 0 or future_frames < 0:
        raise ValueError("past_frames/future_frames must be >= 0")
    if traj.shape != velocities.shape:
        raise ValueError("traj and velocities must have the same shape")

    num_steps, num_agents, _ = traj.shape
    if num_agents == 0:
        raise ValueError("trajectory has no agents")

    first_anchor = int(past_frames)
    last_anchor = int(num_steps - future_frames)
    anchor_steps = list(range(first_anchor, last_anchor, sample_interval))
    if not anchor_steps:
        raise ValueError(
            "No valid anchor timestep. Reduce past/future window or sample interval."
        )

    size_xy = np.array([float(occupancy_length), float(occupancy_width)], dtype=np.float32)
    cells_x = int(np.floor(size_xy[0] / resolution))
    cells_y = int(np.floor(size_xy[1] / resolution))
    if cells_x <= 0 or cells_y <= 0:
        raise ValueError("occupancy size/resolution yields non-positive grid dimensions")

    scene_origin, scene_size, scene_canvas_shape, scene_center = _compute_scene_canvas(
        traj=traj,
        obstacles=obstacles,
        resolution=resolution,
        occupancy_length=occupancy_length,
        occupancy_width=occupancy_width,
    )
    local_h = int(cells_y)
    local_w = int(cells_x)

    scene_static_map = _build_scene_static_map(
        obstacles=obstacles,
        resolution=resolution,
        agent_radius=agent_radius,
        scene_size=scene_size,
        scene_center=scene_center,
    )

    dynamic_maps: List[torch.Tensor] = []
    current_velocities: List[torch.Tensor] = []
    anchor_centers: List[torch.Tensor] = []
    frame_offsets = list(range(-past_frames, future_frames + 1))

    if center_agent_indices is None:
        center_agent_indices = list(range(num_agents))
    if not center_agent_indices:
        raise ValueError("center_agent_indices must not be empty")

    for center_agent_idx in center_agent_indices:
        if center_agent_idx < 0 or center_agent_idx >= num_agents:
            raise ValueError("center_agent_indices contains out-of-range index")
        agent_dynamic = _build_agent_dynamic_map(
            traj=traj,
            center_agent_idx=center_agent_idx,
            resolution=resolution,
            agent_radius=agent_radius,
            scene_size=scene_size,
            scene_center=scene_center,
            canvas_shape=scene_canvas_shape,
        )
        center_series, velocity_series = _collect_anchor_metadata(
            traj=traj,
            velocities=velocities,
            center_agent_idx=center_agent_idx,
            anchor_steps=anchor_steps,
        )

        dynamic_maps.append(agent_dynamic)
        current_velocities.append(velocity_series)
        anchor_centers.append(center_series)

    return (
        dynamic_maps,
        scene_static_map,
        current_velocities,
        scene_origin,
        (resolution, resolution),
        anchor_steps,
        frame_offsets,
        anchor_centers,
        (local_h, local_w),
    )


def _slice_centered_patch(
    grid: torch.Tensor,
    center_xy: torch.Tensor,
    origin_xy: Tuple[float, float],
    resolution_xy: Tuple[float, float],
    patch_shape: Tuple[int, int],
) -> torch.Tensor:
    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
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
    out[dst_y0:dst_y1, dst_x0:dst_x1] = torch.as_tensor(
        grid[src_y0:src_y1, src_x0:src_x1],
        dtype=torch.float32,
    )
    return out


def build_anchor_local_windows(
    *,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    anchor_steps: List[int],
    frame_offsets: List[int],
    scene_origin: Tuple[float, float],
    occupancy_resolution: Tuple[float, float],
    local_map_shape: Tuple[int, int],
    total_steps: int,
) -> Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]]]:
    """Reconstruct per-anchor local static/dynamic windows from global maps."""
    static_maps: List[List[torch.Tensor]] = []
    dynamic_windows: List[List[List[torch.Tensor]]] = []

    for agent_idx, centers in enumerate(anchor_centers):
        centers_tensor = torch.as_tensor(centers, dtype=torch.float32)
        agent_static_series: List[torch.Tensor] = []
        agent_dynamic_windows: List[List[torch.Tensor]] = []
        agent_dynamic = torch.as_tensor(dynamic_maps[agent_idx], dtype=torch.float32)

        for local_idx, anchor_t in enumerate(anchor_steps):
            center_xy = centers_tensor[local_idx]
            static_local = _slice_centered_patch(
                scene_static_map,
                center_xy,
                scene_origin,
                occupancy_resolution,
                local_map_shape,
            )
            agent_static_series.append(static_local)

            anchor_frames: List[torch.Tensor] = []
            for dt_offset in frame_offsets:
                absolute_t = int(anchor_t + dt_offset)
                if absolute_t < 0 or absolute_t >= int(total_steps):
                    anchor_frames.append(torch.zeros(local_map_shape, dtype=torch.float32))
                    continue
                dynamic_local = _slice_centered_patch(
                    agent_dynamic[absolute_t],
                    center_xy,
                    scene_origin,
                    occupancy_resolution,
                    local_map_shape,
                )
                anchor_frames.append(dynamic_local)

            agent_dynamic_windows.append(anchor_frames)

        static_maps.append(agent_static_series)
        dynamic_windows.append(agent_dynamic_windows)

    return static_maps, dynamic_windows


def data_augmentation(
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    current_velocities: List[torch.Tensor],
) -> Tuple[
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
]:
    """Global-map rollout path currently keeps only the original variant.

    Mirror/rotation variants are returned as empty placeholders to preserve
    existing save-loop structure.
    """
    _ = (scene_static_map, dynamic_maps, current_velocities)
    empty_variant: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]] = (
        torch.empty(0, dtype=torch.float32),
        [],
        [],
    )
    return (
        (
            torch.as_tensor(scene_static_map, dtype=torch.float32).clone(),
            [torch.as_tensor(v, dtype=torch.float32).clone() for v in dynamic_maps],
            [torch.as_tensor(v, dtype=torch.float32).clone() for v in current_velocities],
        ),
        empty_variant,
        empty_variant,
        empty_variant,
        empty_variant,
    )


def build_scene_rollout_data(
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    current_velocities: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> SceneRollOutData:
    if not (len(dynamic_maps) == len(current_velocities) == len(anchor_centers)):
        raise ValueError("dynamic_maps, current_velocities, anchor_centers must have same length")

    static_2d = torch.as_tensor(scene_static_map, dtype=torch.float32)
    if static_2d.ndim != 2:
        raise ValueError("scene_static_map must be 2D")

    if not dynamic_maps:
        raise ValueError("dynamic_maps must not be empty")
    scene_dynamic_maps = torch.stack(
        [torch.as_tensor(m, dtype=torch.float32) for m in dynamic_maps],
        dim=0,
    )
    if scene_dynamic_maps.ndim != 4:
        raise ValueError("dynamic_maps must stack into shape (num_agents, total_time, H, W)")

    if int(scene_dynamic_maps.shape[1]) != int(total_steps):
        raise ValueError("dynamic_maps time dimension must match total_steps")

    agents: Dict[int, AgentRollOutData] = {}

    for agent_idx, (velocity_series, center_series) in enumerate(
        zip(current_velocities, anchor_centers)
    ):
        if velocity_series.shape[0] != len(anchor_steps):
            raise ValueError("velocity series length must match anchor_steps")

        centers_tensor = torch.as_tensor(center_series, dtype=torch.float32)
        if centers_tensor.shape != (len(anchor_steps), 2):
            raise ValueError("center series must have shape (num_anchors, 2)")

        agents[agent_idx] = AgentRollOutData(
            agent_index=agent_idx,
            anchor_times=[int(t) for t in anchor_steps],
            anchor_centers=centers_tensor.clone(),
            current_velocities=torch.as_tensor(velocity_series, dtype=torch.float32).clone(),
        )

    return SceneRollOutData(
        dt=float(dt),
        occupancy_resolution=(float(occupancy_resolution[0]), float(occupancy_resolution[1])),
        occupancy_origin=(float(occupancy_origin[0]), float(occupancy_origin[1])),
        frame_offsets=[int(v) for v in frame_offsets],
        agents=agents,
        scene_static_map=static_2d,
        scene_dynamic_maps=scene_dynamic_maps,
        scene_map_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
        local_map_shape=(int(local_map_shape[0]), int(local_map_shape[1])),
    )


def animate_rollout(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    paths: List[PathSpec],
    static_maps: List[List[np.ndarray]],
    dynamic_past_grids: List[List[np.ndarray]],
    dynamic_future_grids: List[List[np.ndarray]],
    occupancy_origins: List[np.ndarray],
    occupancy_resolution: Tuple[float, float],
    anchor_steps: List[int],
    time_step: float,
    title_prefix: str = "",
) -> None:
    """Animate trajectory and occupancy maps (static + past/future dynamic overlay)."""
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    animation_mod = importlib.import_module("matplotlib.animation")
    patches_mod = importlib.import_module("matplotlib.patches")
    FuncAnimation = animation_mod.FuncAnimation
    Polygon = patches_mod.Polygon

    num_steps, _, _ = traj.shape
    occ_steps = len(anchor_steps)
    if occ_steps == 0:
        raise ValueError("anchor_steps must contain at least one timestep")

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

    scat = ax_traj.scatter(traj[anchor_steps[0], :, 0], traj[anchor_steps[0], :, 1], s=60, c="tab:blue")
    ax_traj.scatter(goals[:, 0], goals[:, 1], s=80, c="tab:red", marker="x")
    traj_time_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, va="top")

    num_occ_maps = len(static_maps)
    if num_occ_maps == 0:
        raise ValueError("static_maps must contain at least one occupancy map")
    if len(dynamic_past_grids) != num_occ_maps or len(dynamic_future_grids) != num_occ_maps:
        raise ValueError("dynamic_past_grids and dynamic_future_grids must match static_maps length")

    occ_cols = int(np.ceil(np.sqrt(num_occ_maps)))
    occ_rows = int(np.ceil(num_occ_maps / occ_cols))
    fig_occ, axes_occ = plt.subplots( \
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
        # Occupancy grids are sampled only at anchor steps; keep the latest
        # available anchor as simulation advances at native timestep.
        anchor_idx = int(np.searchsorted(anchor_steps, sim_step, side="right") - 1)
        anchor_idx = max(0, min(anchor_idx, occ_steps - 1))
        anchor_step = anchor_steps[anchor_idx]

        scat.set_offsets(traj[sim_step])
        traj_time_text.set_text(f"t={sim_step * time_step:.2f}s")
        artists = [scat, traj_time_text]
        for occ_idx, (im_static, time_text_static) in enumerate(zip(static_images, static_time_texts)):
            im_static.set_data(static_maps[occ_idx][anchor_idx])
            time_text_static.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={anchor_step * time_step:.2f}s"
            )
            artists.extend([im_static, time_text_static])
        for occ_idx, (im_overlay, time_text_overlay) in enumerate(zip(overlay_images, overlay_time_texts)):
            overlay_rgb = np.stack(
                [
                    dynamic_past_grids[occ_idx][anchor_idx],
                    dynamic_future_grids[occ_idx][anchor_idx],
                    np.zeros_like(dynamic_past_grids[occ_idx][anchor_idx]),
                ],
                axis=-1,
            )
            im_overlay.set_data(np.clip(overlay_rgb, 0.0, 1.0))
            time_text_overlay.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={anchor_step * time_step:.2f}s"
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ORCA pedestrian rollout.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Display a matplotlib animation of the agents.",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help="Save generated rollouts to the data directory as .pt files.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save rollout .pt files (defaults to ../data)",
    )

    parser.add_argument(
        "--disable-data-aug",
        action="store_true",
        help="Disable occupancy data augmentation (mirror + rotations).",
    )
    parser.add_argument(
        "--occ-sample-interval",
        type=int,
        default=1,
        help="Anchor timestep interval for occupancy generation.",
    )
    parser.add_argument(
        "--occ-past-frames",
        type=int,
        default=16,
        help="Past frame count rendered on each anchor occupancy grid.",
    )
    parser.add_argument(
        "--occ-future-frames",
        type=int,
        default=16,
        help="Future frame count rendered on each anchor occupancy grid.",
    )
    parser.add_argument(
        "--save-every-scene",
        action="store_true",
        help="Save each scene immediately as its own .pt file.",
    )
    return parser


def _resolve_data_dir(data_dir_arg: str | None) -> str:
    if data_dir_arg is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    return os.path.abspath(os.path.expanduser(data_dir_arg))


def _compute_variant_sets(
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    current_velocities: List[torch.Tensor],
    data_aug_enabled: bool,
) -> Dict[str, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]]:
    if data_aug_enabled:
        (
            original,
            mirrored,
            rot90,
            rot180,
            rot270,
        ) = data_augmentation(scene_static_map, dynamic_maps, current_velocities)
        return {
            "orig": original,
            "mirror": mirrored,
            "rot90": rot90,
            "rot180": rot180,
            "rot270": rot270,
        }

    empty_variant: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]] = (
        torch.empty(0, dtype=torch.float32),
        [],
        [],
    )
    return {
        "orig": (
            torch.as_tensor(scene_static_map, dtype=torch.float32).clone(),
            [torch.as_tensor(v, dtype=torch.float32).clone() for v in dynamic_maps],
            [torch.as_tensor(v, dtype=torch.float32).clone() for v in current_velocities],
        ),
        "mirror": empty_variant,
        "rot90": empty_variant,
        "rot180": empty_variant,
        "rot270": empty_variant,
    }


def _build_rollout_data_from_variant(
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    variant: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> SceneRollOutData:
    scene_static_map, dynamic_maps, current_velocities = variant
    return build_scene_rollout_data(
        dt=dt,
        occupancy_resolution=occupancy_resolution,
        occupancy_origin=occupancy_origin,
        frame_offsets=frame_offsets,
        anchor_steps=anchor_steps,
        total_steps=total_steps,
        scene_static_map=scene_static_map,
        dynamic_maps=dynamic_maps,
        current_velocities=current_velocities,
        anchor_centers=anchor_centers,
        scene_map_origin=scene_map_origin,
        local_map_shape=local_map_shape,
    )


def _variant_names(data_aug_enabled: bool) -> List[str]:
    names = ["orig"]
    if data_aug_enabled:
        names.extend(["mirror", "rot90", "rot180", "rot270"])
    return names


def _append_scene_rollouts_to_template(
    template_rollouts: Dict[str, List[SceneRollOutData]],
    variants: Dict[str, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]],
    *,
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
    data_aug_enabled: bool,
) -> None:
    for name in _variant_names(data_aug_enabled):
        rollout = _build_rollout_data_from_variant(
            dt=dt,
            occupancy_resolution=occupancy_resolution,
            occupancy_origin=occupancy_origin,
            frame_offsets=frame_offsets,
            anchor_steps=anchor_steps,
            total_steps=total_steps,
            variant=variants[name],
            anchor_centers=anchor_centers,
            scene_map_origin=scene_map_origin,
            local_map_shape=local_map_shape,
        )
        template_rollouts[name].append(rollout)


def _save_template_rollouts(
    *,
    data_dir: str,
    template_name: str,
    template_rollouts: Dict[str, List[SceneRollOutData]],
    data_aug_enabled: bool,
) -> None:
    for name in _variant_names(data_aug_enabled):
        file_name = f"rollout_{template_name}_{name}.pt"
        payload = RollOutData(scenes=template_rollouts[name])
        data_path = os.path.join(data_dir, file_name)
        torch.save(payload, data_path)
        print(f"saved template rollout data: {data_path} ({len(payload.scenes)} scenes)")


def _save_scene_rollouts(
    *,
    data_dir: str,
    template_name: str,
    scene_index: int,
    variants: Dict[str, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]],
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
    data_aug_enabled: bool,
) -> None:
    for name in _variant_names(data_aug_enabled):
        scene_payload = _build_rollout_data_from_variant(
            dt=dt,
            occupancy_resolution=occupancy_resolution,
            occupancy_origin=occupancy_origin,
            frame_offsets=frame_offsets,
            anchor_steps=anchor_steps,
            total_steps=total_steps,
            variant=variants[name],
            anchor_centers=anchor_centers,
            scene_map_origin=scene_map_origin,
            local_map_shape=local_map_shape,
        )
        payload = RollOutData(scenes=[scene_payload])
        file_name = f"rollout_{template_name}_scene{scene_index:05d}_{name}.pt"
        data_path = os.path.join(data_dir, file_name)
        torch.save(payload, data_path)
        print(f"saved scene rollout: {data_path}")


def _prepare_animation_grids(
    static_maps: List[List[torch.Tensor]],
) -> List[List[np.ndarray]]:
    return [
        [grid.detach().cpu().numpy() for grid in per_center_static]
        for per_center_static in static_maps
    ]


def _prepare_past_future_dynamic_grids(
    dynamic_windows: List[List[List[torch.Tensor]]],
    frame_offsets: List[int],
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    past_grids: List[List[np.ndarray]] = []
    future_grids: List[List[np.ndarray]] = []

    for agent_windows in dynamic_windows:
        agent_past: List[np.ndarray] = []
        agent_future: List[np.ndarray] = []
        for anchor_frames in agent_windows:
            if not anchor_frames:
                continue
            if len(anchor_frames) != len(frame_offsets):
                raise ValueError("anchor frame count must match frame_offsets length")

            frame_tensors = [torch.as_tensor(frame, dtype=torch.float32) for frame in anchor_frames]
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


def _print_scene_occupancy_summary(
    scene_index: int,
    traj: np.ndarray,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    variants: Dict[str, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]],
    anchor_steps: List[int],
) -> None:
    for i, pos in enumerate(traj[-1]):
        print(
            f"scene[{scene_index}] agent[{i}] final position: "
            f"({pos[0]:.2f}, {pos[1]:.2f})"
        )

    dynamic_original = variants["orig"][1]
    velocity_original = variants["orig"][2]
    dynamic_shape = tuple(dynamic_maps[0].shape) if dynamic_maps else ()
    print(
        f"scene[{scene_index}] occupancy generated: "
        f"orig={len(variants['orig'][1])}, mirror={len(variants['mirror'][1])}, "
        f"rot90={len(variants['rot90'][1])}, rot180={len(variants['rot180'][1])}, "
        f"rot270={len(variants['rot270'][1])} maps; "
        f"{len(anchor_steps)} anchors each (sampled), "
        f"global static shape {tuple(scene_static_map.shape)}, "
        f"global dynamic shape {dynamic_shape}, "
        f"velocity shape {velocity_original[0].shape}"
    )


def main() -> None:
    """Run an ORCA rollout with optional animation and occupancy-map generation."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    SAVE_ROLLOUTS = bool(args.save_rollouts)

    DATA_DIR = _resolve_data_dir(args.data_dir)

    if SAVE_ROLLOUTS:
        os.makedirs(DATA_DIR, exist_ok=True)

    ANIMATE = bool(args.animate)
    DATA_AUG_ENABLED = not bool(args.disable_data_aug)
    if DATA_AUG_ENABLED:
        print(
            "Global-map rollout format currently saves only the original variant; "
            "data augmentation variants are disabled."
        )
        DATA_AUG_ENABLED = False

    # ORCASim configuration constants
    TIME_STEP = 0.1
    NUM_STEPS = 150
    NEIGHBOR_DIST = 2.0
    MAX_NEIGHBORS = 5
    TIME_HORIZON = 3.0
    TIME_HORIZON_OBST = 5.0
    RADIUS = 0.3
    MAX_SPEED = 5.0
    GOAL_TOLERANCE = 0.2
    PATH_GOAL_SWITCH_TOLERANCE = 3.0
    PATH_SEGMENT_REMAINING_SWITCH_RATIO = 0.05
    PREF_VELOCITY_NOISE_STD = 0.02
    PREF_VELOCITY_NOISE_INTERVAL = 3
    PREF_VELOCITY_NOISE_SEED = 0
    # Occupancy settings
    OCC_RESOLUTION = 0.1
    OCC_AGENT_RADIUS = 0.2
    OCC_LENGTH = 12.8
    OCC_WIDTH = 12.8

    # premade templates are provided by `src.templates.default_templates()`
    rollout_setting = RollOutSetting(
        templates=default_templates(),
        # templates=test_templates(),
        mirror=False,
        rotate=False,
        name="default",
    )

    global_scene_index = 0
    for tpl in rollout_setting.templates:
        template_name = tpl.get_name()
        template_rollouts: Dict[str, List[SceneRollOutData]] = {
            "orig": [],
            "mirror": [],
            "rot90": [],
            "rot180": [],
            "rot270": [],
        }
        scenes = tpl.generate()
        print(
            f"generated {len(scenes)} scenes from {tpl.__class__.__name__} "
            f"(name={template_name}, num_levels={int(tpl.num_levels)})"
        )

        for local_idx, scene in enumerate(scenes):
            scene_index = global_scene_index
            orca_sim = ORCASim(
                scene=scene,
                time_step=TIME_STEP,
                neighbor_dist=NEIGHBOR_DIST,
                max_neighbors=MAX_NEIGHBORS,
                time_horizon=TIME_HORIZON,
                time_horizon_obst=TIME_HORIZON_OBST,
                radius=RADIUS,
                max_speed=MAX_SPEED,
                goal_tolerance=GOAL_TOLERANCE,
                path_goal_switch_tolerance=PATH_GOAL_SWITCH_TOLERANCE,
                path_segment_remaining_switch_ratio=PATH_SEGMENT_REMAINING_SWITCH_RATIO,
                region_pair_seed=scene_index,
                pref_velocity_noise_std=PREF_VELOCITY_NOISE_STD,
                pref_velocity_noise_interval=PREF_VELOCITY_NOISE_INTERVAL,
                pref_velocity_noise_seed=PREF_VELOCITY_NOISE_SEED + scene_index,
            )
            min_steps_for_occupancy = int(args.occ_past_frames) + int(args.occ_future_frames) + 1
            traj, vel_traj = orca_sim.simulate(
                steps=NUM_STEPS,
                min_steps=min_steps_for_occupancy,
                stop_on_goal=True,
                return_velocities=True,
            )
            goals = np.array([agent.goal for agent in scene.agents], dtype=np.float32)
            required_steps = int(args.occ_past_frames) + int(args.occ_future_frames) + 1
            if traj.shape[0] < required_steps:
                print(
                    f"scene[{scene_index}] skipped: steps={traj.shape[0]} < required "
                    f"{required_steps} (past={int(args.occ_past_frames)}, "
                    f"future={int(args.occ_future_frames)})"
                )
                global_scene_index += 1
                continue

            (
                dynamic_maps,
                scene_static_map,
                current_velocities,
                scene_map_origin,
                occupancy_resolution,
                anchor_steps,
                frame_offsets,
                anchor_centers,
                local_map_shape,
            ) = build_agent_centric_occupancy_sequences(
                traj=traj,
                velocities=vel_traj,
                obstacles=scene.obstacles,
                resolution=OCC_RESOLUTION,
                agent_radius=OCC_AGENT_RADIUS,
                occupancy_width=OCC_WIDTH,
                occupancy_length=OCC_LENGTH,
                sample_interval=int(args.occ_sample_interval),
                past_frames=int(args.occ_past_frames),
                future_frames=int(args.occ_future_frames),
            )

            variants = _compute_variant_sets(
                scene_static_map,
                dynamic_maps,
                current_velocities,
                DATA_AUG_ENABLED,
            )

            if SAVE_ROLLOUTS:
                if args.save_every_scene:
                    _save_scene_rollouts(
                        data_dir=DATA_DIR,
                        template_name=template_name,
                        scene_index=scene_index,
                        variants=variants,
                        dt=TIME_STEP,
                        occupancy_resolution=occupancy_resolution,
                        occupancy_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
                        frame_offsets=frame_offsets,
                        anchor_steps=anchor_steps,
                        total_steps=int(traj.shape[0]),
                        anchor_centers=anchor_centers,
                        scene_map_origin=scene_map_origin,
                        local_map_shape=local_map_shape,
                        data_aug_enabled=DATA_AUG_ENABLED,
                    )
                else:
                    _append_scene_rollouts_to_template(
                        template_rollouts=template_rollouts,
                        variants=variants,
                        dt=TIME_STEP,
                        occupancy_resolution=occupancy_resolution,
                        occupancy_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
                        frame_offsets=frame_offsets,
                        anchor_steps=anchor_steps,
                        total_steps=int(traj.shape[0]),
                        anchor_centers=anchor_centers,
                        scene_map_origin=scene_map_origin,
                        local_map_shape=local_map_shape,
                        data_aug_enabled=DATA_AUG_ENABLED,
                    )
                    if DATA_AUG_ENABLED:
                        print(
                            f"scene[{scene_index}] queued split rollout data for template "
                            f"{template_name} (orig/mirror/rot90/rot180/rot270)"
                        )
                    else:
                        print(
                            f"scene[{scene_index}] queued rollout data for template "
                            f"{template_name} (orig only, data aug disabled)"
                        )

            print(
                f"scene[{scene_index}] startup spawn: total agents={traj.shape[1]}, "
                f"steps={traj.shape[0]}"
            )
            
            title_prefix = f"{tpl.__class__.__name__} {local_idx + 1}/{len(scenes)}"
            if ANIMATE:
                static_maps, dynamic_windows = build_anchor_local_windows(
                    scene_static_map=scene_static_map,
                    dynamic_maps=dynamic_maps,
                    anchor_centers=anchor_centers,
                    anchor_steps=anchor_steps,
                    frame_offsets=frame_offsets,
                    scene_origin=scene_map_origin,
                    occupancy_resolution=occupancy_resolution,
                    local_map_shape=local_map_shape,
                    total_steps=int(traj.shape[0]),
                )
                static_maps_np = _prepare_animation_grids(static_maps)
                dynamic_past_grids_np, dynamic_future_grids_np = _prepare_past_future_dynamic_grids(
                    dynamic_windows,
                    frame_offsets,
                )

                local_origins = [
                    np.array(
                        [
                            -0.5 * float(local_map_shape[1]) * float(occupancy_resolution[0]),
                            -0.5 * float(local_map_shape[0]) * float(occupancy_resolution[1]),
                        ],
                        dtype=np.float32,
                    )
                    for _ in range(len(static_maps_np))
                ]

                animate_rollout(
                    traj=traj,
                    goals=goals,
                    obstacles=scene.obstacles,
                    paths=scene.paths,
                    static_maps=static_maps_np,
                    dynamic_past_grids=dynamic_past_grids_np,
                    dynamic_future_grids=dynamic_future_grids_np,
                    occupancy_origins=local_origins,
                    occupancy_resolution=occupancy_resolution,
                    anchor_steps=anchor_steps,
                    time_step=TIME_STEP,
                    title_prefix=title_prefix,
                )
            else:
                _print_scene_occupancy_summary(
                    scene_index=scene_index,
                    traj=traj,
                    scene_static_map=scene_static_map,
                    dynamic_maps=dynamic_maps,
                    variants=variants,
                    anchor_steps=anchor_steps,
                )

            global_scene_index += 1

        if SAVE_ROLLOUTS and (not args.save_every_scene) and len(template_rollouts["orig"]) > 0:
            _save_template_rollouts(
                data_dir=DATA_DIR,
                template_name=template_name,
                template_rollouts=template_rollouts,
                data_aug_enabled=DATA_AUG_ENABLED,
            )


if __name__ == "__main__":
    main()
