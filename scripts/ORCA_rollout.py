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
from src.rollout_data import AgentRollOutData, AnchorRollOutData, RollOutData
from src.rollout_setting import RollOutSetting
from src.templates import default_templates, test_templates


# RollOutSetting is defined in src.rollout_setting

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
    List[List[List[torch.Tensor]]],
    List[List[torch.Tensor]],
    List[torch.Tensor],
    List[np.ndarray],
    Tuple[float, float],
    List[int],
    List[int],
]:
    """Build agent-centric occupancy windows for each centered agent.

    For each agent and anchor timestep, this function creates:
    - one static map centered at the agent position at the anchor,
    - a list of occupancy frames over temporal offsets [t-past, t+future],
      where only other agents are rasterized as dynamic occupancy,
    - the centered agent velocity at the anchor timestep.

    Output structure:
    - dynamic_windows[agent][anchor][offset] -> dynamic occupancy frame.
    - static_maps[agent][anchor] -> static map at the same anchor center.
    - current_velocities[agent][anchor] -> ego velocity at that anchor.

    Rasterization is delegated to `Occupancy2d` with per-anchor center offsets,
    so inputs remain in world coordinates.
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

    static_maps: List[List[torch.Tensor]] = []
    dynamic_windows: List[List[List[torch.Tensor]]] = []
    current_velocities: List[torch.Tensor] = []
    origins: List[np.ndarray] = []
    frame_offsets = list(range(-past_frames, future_frames + 1))

    if center_agent_indices is None:
        center_agent_indices = list(range(num_agents))
    if not center_agent_indices:
        raise ValueError("center_agent_indices must not be empty")

    for center_agent_idx in center_agent_indices:
        if center_agent_idx < 0 or center_agent_idx >= num_agents:
            raise ValueError("center_agent_indices contains out-of-range index")
        sequence_windows: List[List[torch.Tensor]] = []
        static_series: List[torch.Tensor] = []
        velocity_series = torch.zeros((len(anchor_steps), 2), dtype=torch.float32)
        other_agent_indices = [idx for idx in range(num_agents) if idx != center_agent_idx]

        for anchor_idx, anchor_t in enumerate(anchor_steps):
            ego_pos = traj[anchor_t, center_agent_idx].astype(np.float32)

            # Build a static-only occupancy map at this anchor center.
            static_occ = Occupancy2d(
                resolution=(resolution, resolution),
                size=(float(size_xy[0]), float(size_xy[1])),
                trajectory=None,
                static_obstacles=obstacles,
                agent_radius=agent_radius,
            )
            static_grid = static_occ.generate(center_offset=ego_pos)[0].to(dtype=torch.float32)

            window_start = anchor_t - past_frames
            window_end = anchor_t + future_frames + 1
            if other_agent_indices:
                other_traj = traj[window_start:window_end, other_agent_indices].astype(np.float32)
            else:
                other_traj = None

            dynamic_occ = Occupancy2d(
                resolution=(resolution, resolution),
                size=(float(size_xy[0]), float(size_xy[1])),
                trajectory=other_traj,
                static_obstacles=None,
                agent_radius=agent_radius,
            )
            generated_frames = dynamic_occ.generate(center_offset=ego_pos)
            if other_traj is None:
                # Occupancy2d returns one static-only frame when trajectory is None.
                # Repeat it so each anchor still has a full temporal window.
                base_grid = generated_frames[0].to(dtype=torch.float32)
                anchor_frames = [base_grid.clone() for _ in frame_offsets]
            else:
                anchor_frames = [grid.to(dtype=torch.float32) for grid in generated_frames]

            sequence_windows.append(anchor_frames)
            static_series.append(static_grid)
            velocity_series[anchor_idx] = torch.as_tensor(
                velocities[anchor_t, center_agent_idx], dtype=torch.float32
            )

        dynamic_windows.append(sequence_windows)
        static_maps.append(static_series)
        current_velocities.append(velocity_series)
        origins.append(np.array([-0.5 * size_xy[0], -0.5 * size_xy[1]], dtype=np.float32))

    return (
        dynamic_windows,
        static_maps,
        current_velocities,
        origins,
        (resolution, resolution),
        anchor_steps,
        frame_offsets,
    )


def collapse_windows_to_grids(
    dynamic_windows: List[List[List[torch.Tensor]]],
) -> List[List[torch.Tensor]]:
    """Collapse per-anchor windows to one frame via OR over temporal offsets.

    This is used by animation/debug paths that expect one grid per anchor.
    """
    collapsed: List[List[torch.Tensor]] = []
    for agent_windows in dynamic_windows:
        agent_seq: List[torch.Tensor] = []
        for anchor_frames in agent_windows:
            if not anchor_frames:
                continue
            stacked = torch.stack([torch.as_tensor(frame) for frame in anchor_frames], dim=0)
            agent_seq.append(stacked.amax(dim=0))
        collapsed.append(agent_seq)
    return collapsed


def collapse_static_series(
    static_series_per_agent: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """Collapse per-anchor static maps into one map per agent for display compatibility."""
    collapsed: List[torch.Tensor] = []
    for static_series in static_series_per_agent:
        if not static_series:
            continue
        stacked = torch.stack([torch.as_tensor(frame) for frame in static_series], dim=0)
        collapsed.append(stacked.amax(dim=0))
    return collapsed


def data_augmentation(
    static_maps: List[List[torch.Tensor]],
    dynamic_windows: List[List[List[torch.Tensor]]],
    current_velocities: List[torch.Tensor],
) -> Tuple[
    Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
    Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
    Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
    Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
    Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
]:
    """Return static+dynamic variants: original, mirror, rot90, rot180, rot270.

    Rotations are clockwise around the grid center.
    """

    if len(static_maps) != len(dynamic_windows):
        raise ValueError("static_maps and dynamic_windows must have the same length")
    if len(current_velocities) != len(dynamic_windows):
        raise ValueError("current_velocities and dynamic_windows must have the same length")

    def _tensor_to_windows(seq_tensor: torch.Tensor) -> List[List[torch.Tensor]]:
        return [
            [frame.clone() for frame in anchor_tensor]
            for anchor_tensor in seq_tensor
        ]

    original_static: List[List[torch.Tensor]] = []
    original_dynamic: List[List[List[torch.Tensor]]] = []
    original_velocity: List[torch.Tensor] = []
    mirrored_static: List[List[torch.Tensor]] = []
    mirrored_dynamic: List[List[List[torch.Tensor]]] = []
    mirrored_velocity: List[torch.Tensor] = []
    rot90_static: List[List[torch.Tensor]] = []
    rot90_dynamic: List[List[List[torch.Tensor]]] = []
    rot90_velocity: List[torch.Tensor] = []
    rot180_static: List[List[torch.Tensor]] = []
    rot180_dynamic: List[List[List[torch.Tensor]]] = []
    rot180_velocity: List[torch.Tensor] = []
    rot270_static: List[List[torch.Tensor]] = []
    rot270_dynamic: List[List[List[torch.Tensor]]] = []
    rot270_velocity: List[torch.Tensor] = []

    for static_series, sequence_windows, velocity_series in zip(static_maps, dynamic_windows, current_velocities):
        if not sequence_windows:
            continue

        # Stack once so transforms are applied consistently to all timesteps.
        static_tensor = torch.stack([torch.as_tensor(frame) for frame in static_series], dim=0)
        seq_tensor = torch.stack(
            [
                torch.stack([torch.as_tensor(frame) for frame in anchor_frames], dim=0)
                for anchor_frames in sequence_windows
            ],
            dim=0,
        )

        original_static.append([frame.clone() for frame in static_tensor])
        original_dynamic.append(_tensor_to_windows(seq_tensor))
        original_velocity.append(torch.as_tensor(velocity_series, dtype=torch.float32).clone())

        mirrored_static.append([frame.clone() for frame in torch.flip(static_tensor, dims=(-1,))])
        mirrored_dynamic.append(_tensor_to_windows(torch.flip(seq_tensor, dims=(-1,))))
        mirrored_vel = torch.as_tensor(velocity_series, dtype=torch.float32).clone()
        mirrored_vel[:, 0] *= -1.0
        mirrored_velocity.append(mirrored_vel)

        rot90_static.append([frame.clone() for frame in torch.rot90(static_tensor, k=-1, dims=(-2, -1))])
        rot90_dynamic.append(_tensor_to_windows(torch.rot90(seq_tensor, k=-1, dims=(-2, -1))))
        rot90_vel = torch.as_tensor(velocity_series, dtype=torch.float32).clone()
        rot90_vel = torch.stack([rot90_vel[:, 1], -rot90_vel[:, 0]], dim=1)
        rot90_velocity.append(rot90_vel)

        rot180_static.append([frame.clone() for frame in torch.rot90(static_tensor, k=2, dims=(-2, -1))])
        rot180_dynamic.append(_tensor_to_windows(torch.rot90(seq_tensor, k=2, dims=(-2, -1))))
        rot180_velocity.append(-torch.as_tensor(velocity_series, dtype=torch.float32).clone())

        rot270_static.append([frame.clone() for frame in torch.rot90(static_tensor, k=-3, dims=(-2, -1))])
        rot270_dynamic.append(_tensor_to_windows(torch.rot90(seq_tensor, k=-3, dims=(-2, -1))))
        rot270_vel = torch.as_tensor(velocity_series, dtype=torch.float32).clone()
        rot270_vel = torch.stack([-rot270_vel[:, 1], rot270_vel[:, 0]], dim=1)
        rot270_velocity.append(rot270_vel)

    return (
        (original_static, original_dynamic, original_velocity),
        (mirrored_static, mirrored_dynamic, mirrored_velocity),
        (rot90_static, rot90_dynamic, rot90_velocity),
        (rot180_static, rot180_dynamic, rot180_velocity),
        (rot270_static, rot270_dynamic, rot270_velocity),
    )


def build_scene_rollout_data(
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    static_maps: List[List[torch.Tensor]],
    dynamic_windows: List[List[List[torch.Tensor]]],
    current_velocities: List[torch.Tensor],
) -> RollOutData:
    if not (len(static_maps) == len(dynamic_windows) == len(current_velocities)):
        raise ValueError("static_maps, dynamic_windows, current_velocities must have same length")

    agents: Dict[int, AgentRollOutData] = {}
    for agent_idx, (static_series, agent_windows, velocity_series) in enumerate(
        zip(static_maps, dynamic_windows, current_velocities)
    ):
        if len(static_series) != len(anchor_steps):
            raise ValueError("static series length must match anchor_steps")
        if len(agent_windows) != len(anchor_steps):
            raise ValueError("agent_windows length must match anchor_steps")
        if velocity_series.shape[0] != len(anchor_steps):
            raise ValueError("velocity series length must match anchor_steps")

        anchor_map: Dict[int, AnchorRollOutData] = {}
        for local_idx, anchor_t in enumerate(anchor_steps):
            anchor_map[int(anchor_t)] = AnchorRollOutData(
                anchor_time=int(anchor_t),
                static_map=torch.as_tensor(static_series[local_idx], dtype=torch.float32).clone(),
                current_velocity=torch.as_tensor(velocity_series[local_idx], dtype=torch.float32),
                frames=[torch.as_tensor(frame, dtype=torch.float32).clone() for frame in agent_windows[local_idx]],
            )

        agents[agent_idx] = AgentRollOutData(
            agent_index=agent_idx,
            anchors=anchor_map,
        )

    return RollOutData(
        dt=float(dt),
        occupancy_resolution=(float(occupancy_resolution[0]), float(occupancy_resolution[1])),
        occupancy_origin=(float(occupancy_origin[0]), float(occupancy_origin[1])),
        frame_offsets=[int(v) for v in frame_offsets],
        agents=agents,
    )


def animate_rollout(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    paths: List[PathSpec],
    occupancy_grids: List[List[np.ndarray]],
    static_maps: List[List[np.ndarray]],
    dynamic_grids: List[List[np.ndarray]],
    occupancy_origins: List[np.ndarray],
    occupancy_resolution: Tuple[float, float],
    anchor_steps: List[int],
    time_step: float,
    title_prefix: str = "",
) -> None:
    """Animate trajectory and per-anchor full/dynamic/static occupancy maps."""
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
    occ_title = "Occupancy Map"
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

    num_occ_maps = len(occupancy_grids)
    if num_occ_maps == 0:
        raise ValueError("occupancy_grids must contain at least one occupancy map")

    occ_cols = int(np.ceil(np.sqrt(num_occ_maps)))
    occ_rows = int(np.ceil(num_occ_maps / occ_cols))
    fig_occ, axes_occ = plt.subplots(
        occ_rows,
        occ_cols,
        figsize=(4.0 * occ_cols, 4.0 * occ_rows),
        squeeze=False,
    )
    fig_occ.suptitle(occ_title)

    flat_axes = axes_occ.flatten()
    occ_images = []
    occ_time_texts = []
    res_x, res_y = occupancy_resolution

    for occ_idx in range(num_occ_maps):
        ax_occ = flat_axes[occ_idx]
        first_grid = occupancy_grids[occ_idx][0]
        cells_y, cells_x = first_grid.shape
        origin = occupancy_origins[occ_idx]
        extent = (
            float(origin[0]),
            float(origin[0] + cells_x * res_x),
            float(origin[1]),
            float(origin[1] + cells_y * res_y),
        )
        im = ax_occ.imshow(
            first_grid,
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=extent,
            interpolation="nearest",
        )
        ax_occ.set_title(f"Occ #{occ_idx}")
        ax_occ.set_xlabel("X")
        ax_occ.set_ylabel("Y")
        ax_occ.set_aspect("equal", adjustable="box")
        time_text = ax_occ.text(0.02, 0.98, "", transform=ax_occ.transAxes, va="top")
        occ_images.append(im)
        occ_time_texts.append(time_text)

    for ax_occ in flat_axes[num_occ_maps:]:
        ax_occ.axis("off")

    fig_dyn, axes_dyn = plt.subplots(
        occ_rows,
        occ_cols,
        figsize=(4.0 * occ_cols, 4.0 * occ_rows),
        squeeze=False,
    )
    dyn_title = "Dynamic Occupancy Map"
    if title_prefix:
        dyn_title = f"{title_prefix} - {dyn_title}"
    fig_dyn.suptitle(dyn_title)

    flat_dyn_axes = axes_dyn.flatten()
    dyn_images = []
    dyn_time_texts = []
    for occ_idx in range(num_occ_maps):
        ax_dyn = flat_dyn_axes[occ_idx]
        first_dyn = dynamic_grids[occ_idx][0]
        cells_y, cells_x = first_dyn.shape
        origin = occupancy_origins[occ_idx]
        extent = (
            float(origin[0]),
            float(origin[0] + cells_x * res_x),
            float(origin[1]),
            float(origin[1] + cells_y * res_y),
        )
        im_dyn = ax_dyn.imshow(
            first_dyn,
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=extent,
            interpolation="nearest",
        )
        ax_dyn.set_title(f"Dynamic #{occ_idx}")
        ax_dyn.set_xlabel("X")
        ax_dyn.set_ylabel("Y")
        ax_dyn.set_aspect("equal", adjustable="box")
        time_text = ax_dyn.text(0.02, 0.98, "", transform=ax_dyn.transAxes, va="top")
        dyn_images.append(im_dyn)
        dyn_time_texts.append(time_text)

    for ax_dyn in flat_dyn_axes[num_occ_maps:]:
        ax_dyn.axis("off")

    fig_static, axes_static = plt.subplots(
        occ_rows,
        occ_cols,
        figsize=(4.0 * occ_cols, 4.0 * occ_rows),
        squeeze=False,
    )
    static_title = "Static Occupancy Map"
    if title_prefix:
        static_title = f"{title_prefix} - {static_title}"
    fig_static.suptitle(static_title)

    flat_static_axes = axes_static.flatten()
    static_images = []
    static_time_texts = []
    for occ_idx in range(num_occ_maps):
        ax_static = flat_static_axes[occ_idx]
        first_static = static_maps[occ_idx][0]
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
        ax_static.set_title(f"Static #{occ_idx}")
        ax_static.set_xlabel("X")
        ax_static.set_ylabel("Y")
        ax_static.set_aspect("equal", adjustable="box")
        time_text = ax_static.text(0.02, 0.98, "", transform=ax_static.transAxes, va="top")
        static_images.append(im_static)
        static_time_texts.append(time_text)

    for ax_static in flat_static_axes[num_occ_maps:]:
        ax_static.axis("off")

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
        for occ_idx, (im, time_text) in enumerate(zip(occ_images, occ_time_texts)):
            im.set_data(occupancy_grids[occ_idx][anchor_idx])
            time_text.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={anchor_step * time_step:.2f}s"
            )
            artists.extend([im, time_text])
        for occ_idx, (im_dyn, time_text_dyn) in enumerate(zip(dyn_images, dyn_time_texts)):
            im_dyn.set_data(dynamic_grids[occ_idx][anchor_idx])
            time_text_dyn.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={anchor_step * time_step:.2f}s"
            )
            artists.extend([im_dyn, time_text_dyn])
        for occ_idx, (im_static, time_text_static) in enumerate(zip(static_images, static_time_texts)):
            im_static.set_data(static_maps[occ_idx][anchor_idx])
            time_text_static.set_text(
                f"sim t={sim_step * time_step:.2f}s\\nocc@t={anchor_step * time_step:.2f}s"
            )
            artists.extend([im_static, time_text_static])
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
    anim_dyn = FuncAnimation(
        fig_dyn,
        update,
        frames=num_steps,
        interval=time_step * 1000,
        blit=False,
    )
    anim_static = FuncAnimation(
        fig_static,
        update,
        frames=num_steps,
        interval=time_step * 1000,
        blit=False,
    )

    plt.tight_layout()
    plt.show()

    _ = (anim_traj, anim_occ, anim_dyn, anim_static)


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
        "--num-levels",
        type=int,
        default=5,
        help="Number of levels to generate per template (default: 5)",
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
        "--occ-agent-chunk-size",
        type=int,
        default=0,
        help=(
            "Process centered agents in chunks to reduce peak RAM. "
            "0 means process all agents at once."
        ),
    )
    parser.add_argument(
        "--save-per-scene",
        action="store_true",
        help=(
            "When saving rollouts, write scene/chunk files immediately instead of "
            "buffering all scenes per template in memory."
        ),
    )
    return parser


def _resolve_data_dir(data_dir_arg: str | None) -> str:
    if data_dir_arg is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    return os.path.abspath(os.path.expanduser(data_dir_arg))


def _compute_variant_sets(
    static_maps: List[List[torch.Tensor]],
    dynamic_windows: List[List[List[torch.Tensor]]],
    current_velocities: List[torch.Tensor],
    data_aug_enabled: bool,
) -> Dict[str, Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]]]:
    if data_aug_enabled:
        (
            original,
            mirrored,
            rot90,
            rot180,
            rot270,
        ) = data_augmentation(static_maps, dynamic_windows, current_velocities)
        return {
            "orig": original,
            "mirror": mirrored,
            "rot90": rot90,
            "rot180": rot180,
            "rot270": rot270,
        }

    empty_variant: Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]] = (
        [],
        [],
        [],
    )
    return {
        "orig": (static_maps, dynamic_windows, current_velocities),
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
    variant: Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]],
) -> RollOutData:
    static_maps, dynamic_windows, current_velocities = variant
    return build_scene_rollout_data(
        dt=dt,
        occupancy_resolution=occupancy_resolution,
        occupancy_origin=occupancy_origin,
        frame_offsets=frame_offsets,
        anchor_steps=anchor_steps,
        static_maps=static_maps,
        dynamic_windows=dynamic_windows,
        current_velocities=current_velocities,
    )


def _variant_names(data_aug_enabled: bool) -> List[str]:
    names = ["orig"]
    if data_aug_enabled:
        names.extend(["mirror", "rot90", "rot180", "rot270"])
    return names


def _append_scene_rollouts_to_template(
    template_rollouts: Dict[str, List[RollOutData]],
    variants: Dict[str, Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]]],
    *,
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    data_aug_enabled: bool,
) -> None:
    for name in _variant_names(data_aug_enabled):
        rollout = _build_rollout_data_from_variant(
            dt=dt,
            occupancy_resolution=occupancy_resolution,
            occupancy_origin=occupancy_origin,
            frame_offsets=frame_offsets,
            anchor_steps=anchor_steps,
            variant=variants[name],
        )
        template_rollouts[name].append(rollout)


def _save_scene_chunk_rollouts(
    *,
    data_dir: str,
    template_name: str,
    scene_index: int,
    chunk_index: int,
    variants: Dict[str, Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]]],
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    data_aug_enabled: bool,
) -> None:
    chunk_tag = f"scene{scene_index:05d}_chunk{chunk_index:03d}"
    for name in _variant_names(data_aug_enabled):
        payload = _build_rollout_data_from_variant(
            dt=dt,
            occupancy_resolution=occupancy_resolution,
            occupancy_origin=occupancy_origin,
            frame_offsets=frame_offsets,
            anchor_steps=anchor_steps,
            variant=variants[name],
        )
        file_name = f"rollout_{template_name}_{chunk_tag}_{name}.pt"
        data_path = os.path.join(data_dir, file_name)
        torch.save([payload], data_path)
        print(f"saved rollout chunk: {data_path} (scene={scene_index})")


def _save_template_rollouts(
    *,
    data_dir: str,
    template_name: str,
    template_rollouts: Dict[str, List[RollOutData]],
    data_aug_enabled: bool,
) -> None:
    for name in _variant_names(data_aug_enabled):
        file_name = f"rollout_{template_name}_{name}.pt"
        payload = template_rollouts[name]
        data_path = os.path.join(data_dir, file_name)
        torch.save(payload, data_path)
        print(f"saved template rollout data: {data_path} ({len(payload)} scenes)")


def _prepare_animation_grids(
    static_maps: List[List[torch.Tensor]],
    dynamic_grids: List[List[torch.Tensor]],
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
    static_maps_np = [
        [grid.detach().cpu().numpy() for grid in per_center_static]
        for per_center_static in static_maps
    ]
    dynamic_grids_np = [
        [grid.detach().cpu().numpy() for grid in per_center_grids]
        for per_center_grids in dynamic_grids
    ]
    occupancy_grids_np = [
        [
            np.clip(static_maps_np[occ_idx][anchor_idx] + dynamic_grids_np[occ_idx][anchor_idx], 0.0, 1.0)
            for anchor_idx in range(len(dynamic_grids_np[occ_idx]))
        ]
        for occ_idx in range(len(dynamic_grids_np))
    ]
    return static_maps_np, dynamic_grids_np, occupancy_grids_np


def _print_scene_occupancy_summary(
    scene_index: int,
    traj: np.ndarray,
    static_maps_display: List[torch.Tensor],
    dynamic_grids: List[List[torch.Tensor]],
    variants: Dict[str, Tuple[List[List[torch.Tensor]], List[List[List[torch.Tensor]]], List[torch.Tensor]]],
) -> None:
    for i, pos in enumerate(traj[-1]):
        print(
            f"scene[{scene_index}] agent[{i}] final position: "
            f"({pos[0]:.2f}, {pos[1]:.2f})"
        )

    dynamic_original = variants["orig"][1]
    velocity_original = variants["orig"][2]
    print(
        f"scene[{scene_index}] occupancy generated: "
        f"orig={len(variants['orig'][1])}, mirror={len(variants['mirror'][1])}, "
        f"rot90={len(variants['rot90'][1])}, rot180={len(variants['rot180'][1])}, "
        f"rot270={len(variants['rot270'][1])} maps; "
        f"{len(dynamic_grids[0])} anchors each (sampled), "
        f"{len(dynamic_original[0][0])} frames per anchor window, "
        f"static shape {static_maps_display[0].shape}, "
        f"dynamic grid shape {dynamic_grids[0][0].shape}, "
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
    NUM_LEVELS_PER_TEMPLATE = int(args.num_levels)

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
        template_rollouts: Dict[str, List[RollOutData]] = {
            "orig": [],
            "mirror": [],
            "rot90": [],
            "rot180": [],
            "rot270": [],
        }
        scenes = tpl.generate(num_levels=NUM_LEVELS_PER_TEMPLATE)
        print(
            f"generated {len(scenes)} scenes from {tpl.__class__.__name__} "
            f"(name={template_name})"
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

            num_agents = int(traj.shape[1])
            requested_chunk_size = int(args.occ_agent_chunk_size)
            if requested_chunk_size <= 0:
                chunk_size = num_agents
            else:
                chunk_size = min(requested_chunk_size, num_agents)

            scene_dynamic_windows: List[List[List[torch.Tensor]]] = []
            scene_static_maps: List[List[torch.Tensor]] = []
            scene_current_velocities: List[torch.Tensor] = []
            scene_occupancy_origin: List[np.ndarray] = []
            scene_anchor_steps: List[int] | None = None
            scene_frame_offsets: List[int] | None = None
            scene_occupancy_resolution: Tuple[float, float] | None = None

            for chunk_start in range(0, num_agents, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_agents)
                center_agent_indices = list(range(chunk_start, chunk_end))
                chunk_index = chunk_start // chunk_size
                (
                    chunk_dynamic_windows,
                    chunk_static_maps,
                    chunk_current_velocities,
                    chunk_occupancy_origin,
                    chunk_occupancy_resolution,
                    chunk_anchor_steps,
                    chunk_frame_offsets,
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
                    center_agent_indices=center_agent_indices,
                )

                if scene_anchor_steps is None:
                    scene_anchor_steps = chunk_anchor_steps
                elif scene_anchor_steps != chunk_anchor_steps:
                    raise ValueError("anchor_steps mismatch across chunks")

                if scene_frame_offsets is None:
                    scene_frame_offsets = chunk_frame_offsets
                elif scene_frame_offsets != chunk_frame_offsets:
                    raise ValueError("frame_offsets mismatch across chunks")

                if scene_occupancy_resolution is None:
                    scene_occupancy_resolution = chunk_occupancy_resolution
                elif scene_occupancy_resolution != chunk_occupancy_resolution:
                    raise ValueError("occupancy_resolution mismatch across chunks")

                scene_dynamic_windows.extend(chunk_dynamic_windows)
                scene_static_maps.extend(chunk_static_maps)
                scene_current_velocities.extend(chunk_current_velocities)
                scene_occupancy_origin.extend(chunk_occupancy_origin)

                if SAVE_ROLLOUTS and args.save_per_scene:
                    chunk_variants = _compute_variant_sets(
                        chunk_static_maps,
                        chunk_dynamic_windows,
                        chunk_current_velocities,
                        DATA_AUG_ENABLED,
                    )
                    chunk_origin_tuple = (
                        float(chunk_occupancy_origin[0][0]),
                        float(chunk_occupancy_origin[0][1]),
                    )
                    _save_scene_chunk_rollouts(
                        data_dir=DATA_DIR,
                        template_name=template_name,
                        scene_index=scene_index,
                        chunk_index=chunk_index,
                        variants=chunk_variants,
                        dt=TIME_STEP,
                        occupancy_resolution=chunk_occupancy_resolution,
                        occupancy_origin=chunk_origin_tuple,
                        frame_offsets=chunk_frame_offsets,
                        anchor_steps=chunk_anchor_steps,
                        data_aug_enabled=DATA_AUG_ENABLED,
                    )

            if scene_anchor_steps is None or scene_frame_offsets is None or scene_occupancy_resolution is None:
                raise RuntimeError("failed to generate occupancy chunks")

            dynamic_windows = scene_dynamic_windows
            static_maps = scene_static_maps
            current_velocities = scene_current_velocities
            occupancy_origin = scene_occupancy_origin
            occupancy_resolution = scene_occupancy_resolution
            anchor_steps = scene_anchor_steps
            frame_offsets = scene_frame_offsets

            dynamic_grids = collapse_windows_to_grids(dynamic_windows)
            static_maps_display = collapse_static_series(static_maps)
            variants = _compute_variant_sets(
                static_maps,
                dynamic_windows,
                current_velocities,
                DATA_AUG_ENABLED,
            )

            if SAVE_ROLLOUTS and not args.save_per_scene:
                _append_scene_rollouts_to_template(
                    template_rollouts=template_rollouts,
                    variants=variants,
                    dt=TIME_STEP,
                    occupancy_resolution=occupancy_resolution,
                    occupancy_origin=(float(occupancy_origin[0][0]), float(occupancy_origin[0][1])),
                    frame_offsets=frame_offsets,
                    anchor_steps=anchor_steps,
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
                static_maps_np, dynamic_grids_np, occupancy_grids_np = _prepare_animation_grids(
                    static_maps,
                    dynamic_grids,
                )

                animate_rollout(
                    traj=traj,
                    goals=goals,
                    obstacles=scene.obstacles,
                    paths=scene.paths,
                    occupancy_grids=occupancy_grids_np,
                    static_maps=static_maps_np,
                    dynamic_grids=dynamic_grids_np,
                    occupancy_origins=occupancy_origin,
                    occupancy_resolution=occupancy_resolution,
                    anchor_steps=anchor_steps,
                    time_step=TIME_STEP,
                    title_prefix=title_prefix,
                )
            else:
                _print_scene_occupancy_summary(
                    scene_index=scene_index,
                    traj=traj,
                    static_maps_display=static_maps_display,
                    dynamic_grids=dynamic_grids,
                    variants=variants,
                )

            global_scene_index += 1

        if SAVE_ROLLOUTS and not args.save_per_scene:
            _save_template_rollouts(
                data_dir=DATA_DIR,
                template_name=template_name,
                template_rollouts=template_rollouts,
                data_aug_enabled=DATA_AUG_ENABLED,
            )


if __name__ == "__main__":
    main()
