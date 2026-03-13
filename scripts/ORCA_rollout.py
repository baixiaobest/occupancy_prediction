from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import os
import sys

# Ensure project root is on sys.path so `from src...` works when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scene import ObstacleSpec, PathSpec
from src.ORCASim import ORCASim
from src.occupancy2d import Occupancy2d
from src.rollout_setting import RollOutSetting
from src.templates import default_templates


OCC_RESOLUTION = 0.1
OCC_MARGIN = 0.2
OCC_AGENT_RADIUS = 0.2


# RollOutSetting is defined in src.rollout_setting


def build_occupancy_maps(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    resolution: float,
    margin: float,
    agent_radius: float,
    occupancy_length: float | None = None,
    occupancy_width: float | None = None,
    occupancy_center: Tuple[float, float] | np.ndarray | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, Tuple[float, float]]:
    """Build per-timestep occupancy grids from rollout trajectory and static obstacles.

    Args:
        traj: Rollout trajectory with shape (T, N, 2).
        goals: Goal positions with shape (N, 2).
        obstacles: Static polygon obstacles.
        resolution: Occupancy grid resolution in meters per cell.
        margin: Padding applied to automatic sizing bounds.
        agent_radius: Agent rasterization radius.
        occupancy_length: Optional map length along the x-axis. If omitted,
            x-size is derived from trajectory/goals/obstacles bounds plus margin.
        occupancy_width: Optional map width along the y-axis. If omitted,
            y-size is derived from trajectory/goals/obstacles bounds plus margin.
        occupancy_center: Optional map center `(x, y)` in world coordinates. If
            omitted, map placement follows current automatic logic.

    Returns:
        A tuple `(grids, origin, resolution_xy)` where `origin` maps grid coordinates
        back to world coordinates.
    """
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

    auto_origin = (min_xy - margin).astype(np.float32)
    auto_upper = (max_xy + margin).astype(np.float32)
    auto_size = auto_upper - auto_origin

    if occupancy_length is not None and occupancy_length <= 0:
        raise ValueError("occupancy_length must be positive when provided")
    if occupancy_width is not None and occupancy_width <= 0:
        raise ValueError("occupancy_width must be positive when provided")

    center_override: np.ndarray | None = None
    if occupancy_center is not None:
        center_arr = np.asarray(occupancy_center, dtype=np.float32)
        if center_arr.shape != (2,):
            raise ValueError("occupancy_center must contain exactly two values (x, y)")
        center_override = center_arr

    center = 0.5 * (min_xy + max_xy)
    size = auto_size.copy()
    origin = auto_origin.copy()

    if occupancy_length is not None:
        size[0] = float(occupancy_length)
        origin[0] = float(center[0] - 0.5 * size[0])

    if occupancy_width is not None:
        size[1] = float(occupancy_width)
        origin[1] = float(center[1] - 0.5 * size[1])

    if center_override is not None:
        origin[0] = float(center_override[0] - 0.5 * size[0])
        origin[1] = float(center_override[1] - 0.5 * size[1])

    shifted_traj = traj - origin[None, None, :]
    shifted_obstacles: List[List[Tuple[float, float]]] = []
    for obstacle in obstacles:
        shifted_obstacles.append(
            [
                (float(vx - origin[0]), float(vy - origin[1]))
                for vx, vy in obstacle.vertices
            ]
        )

    occ2d = Occupancy2d(
        resolution=(resolution, resolution),
        size=(float(size[0]), float(size[1])),
        trajectory=shifted_traj,
        static_obstacles=shifted_obstacles,
        agent_radius=agent_radius,
    )
    grids = [grid.cpu().numpy() for grid in occ2d.generate()]
    return grids, origin, (resolution, resolution)


def animate_rollout(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    paths: List[PathSpec],
    occupancy_grids: List[np.ndarray],
    occupancy_origin: np.ndarray,
    occupancy_resolution: Tuple[float, float],
    time_step: float,
    title_prefix: str = "",
) -> None:
    """Animate trajectory and occupancy map in two synchronized matplotlib windows."""
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    animation_mod = importlib.import_module("matplotlib.animation")
    patches_mod = importlib.import_module("matplotlib.patches")
    FuncAnimation = animation_mod.FuncAnimation
    Polygon = patches_mod.Polygon

    num_steps, _, _ = traj.shape

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

    scat = ax_traj.scatter(traj[0, :, 0], traj[0, :, 1], s=60, c="tab:blue")
    ax_traj.scatter(goals[:, 0], goals[:, 1], s=80, c="tab:red", marker="x")
    traj_time_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, va="top")

    fig_occ, ax_occ = plt.subplots(figsize=(6, 6))
    ax_occ.set_title(occ_title)
    ax_occ.set_xlabel("X")
    ax_occ.set_ylabel("Y")
    ax_occ.set_aspect("equal", adjustable="box")

    first_grid = occupancy_grids[0]
    cells_y, cells_x = first_grid.shape
    res_x, res_y = occupancy_resolution
    extent = (
        float(occupancy_origin[0]),
        float(occupancy_origin[0] + cells_x * res_x),
        float(occupancy_origin[1]),
        float(occupancy_origin[1] + cells_y * res_y),
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
    fig_occ.colorbar(im, ax=ax_occ, fraction=0.046, pad=0.04)
    occ_time_text = ax_occ.text(0.02, 0.98, "", transform=ax_occ.transAxes, va="top")

    def update(frame: int):
        scat.set_offsets(traj[frame])
        traj_time_text.set_text(f"t={frame * time_step:.2f}s")
        im.set_data(occupancy_grids[frame])
        occ_time_text.set_text(f"t={frame * time_step:.2f}s")
        return scat, traj_time_text, im, occ_time_text

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
def main() -> None:
    """Run an ORCA rollout with optional animation and occupancy-map generation."""
    parser = argparse.ArgumentParser(description="Run an ORCA pedestrian rollout.")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Display a matplotlib animation of the agents.",
    )

    args = parser.parse_args()
    # ORCASim configuration constants
    TIME_STEP = args.dt
    NEIGHBOR_DIST = 3.0
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

    # premade templates are provided by `src.templates.default_templates()`
    rollout_setting = RollOutSetting(
        templates=default_templates(),
        mirror=False,
        rotate=False,
        name="default",
    )

    global_scene_index = 0
    for tpl in rollout_setting.templates:
        scenes = tpl.generate(num_levels=5)
        print(f"generated {len(scenes)} scenes from {tpl.__class__.__name__}")

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
            traj = orca_sim.simulate(steps=args.steps, stop_on_goal=True)
            goals = np.array([agent.goal for agent in scene.agents], dtype=np.float32)
            occupancy_grids, occupancy_origin, occupancy_resolution = build_occupancy_maps(
                traj=traj,
                goals=goals,
                obstacles=scene.obstacles,
                resolution=OCC_RESOLUTION,
                margin=OCC_MARGIN,
                agent_radius=OCC_AGENT_RADIUS,
                occupancy_center=(0.0, 0.0),
                occupancy_width=10.0,
                occupancy_length=10.0,
            )

            print(
                f"scene[{scene_index}] startup spawn: total agents={traj.shape[1]}, "
                f"steps={traj.shape[0]}"
            )

            title_prefix = f"{tpl.__class__.__name__} {local_idx + 1}/{len(scenes)}"
            if args.animate:
                animate_rollout(
                    traj=traj,
                    goals=goals,
                    obstacles=scene.obstacles,
                    paths=scene.paths,
                    occupancy_grids=occupancy_grids,
                    occupancy_origin=occupancy_origin,
                    occupancy_resolution=occupancy_resolution,
                    time_step=TIME_STEP,
                    title_prefix=title_prefix,
                )
            else:
                for i, pos in enumerate(traj[-1]):
                    print(
                        f"scene[{scene_index}] agent[{i}] final position: "
                        f"({pos[0]:.2f}, {pos[1]:.2f})"
                    )
                print(
                    f"scene[{scene_index}] occupancy generated: {len(occupancy_grids)} frames, "
                    f"grid shape {occupancy_grids[0].shape}"
                )

            global_scene_index += 1


if __name__ == "__main__":
    main()
