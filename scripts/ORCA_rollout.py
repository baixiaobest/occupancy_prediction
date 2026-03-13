from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import os
import sys

# Ensure project root is on sys.path so `from src...` works when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scene import ObstacleSpec
from src.ORCASim import ORCASim
from src.occupancy2d import Occupancy2d
from src.scene_template import StraightCorridorTemplate


OCC_RESOLUTION = 0.1
OCC_MARGIN = 0.2
OCC_AGENT_RADIUS = 0.2


def build_occupancy_maps(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    resolution: float,
    margin: float,
    agent_radius: float,
) -> Tuple[List[np.ndarray], np.ndarray, Tuple[float, float]]:
    """Build per-timestep occupancy grids from rollout trajectory and static obstacles.

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

    origin = (min_xy - margin).astype(np.float32)
    upper = (max_xy + margin).astype(np.float32)
    size = upper - origin

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


def get_L_shape_scene() -> Scene:
    corridor_obstacles = [
        ObstacleSpec(vertices=[(-15.0, -4.4), (4.4, -4.4), (4.4, -4.0), (-15.0, -4.0)]),
        ObstacleSpec(vertices=[(-15.0, 4.0), (-4.0, 4.0), (-4.0, 4.4), (-15.0, 4.4)]),
        ObstacleSpec(vertices=[(-15.4, -4.4), (-15.0, -4.4), (-15.0, 4.4), (-15.4, 4.4)]),
        ObstacleSpec(vertices=[(4.0, -4.4), (4.4, -4.4), (4.4, 15.4), (4.0, 15.4)]),
        ObstacleSpec(vertices=[(-4.4, 4.4), (-4.0, 4.4), (-4.0, 15.4), (-4.4, 15.4)]),
        ObstacleSpec(vertices=[(-4.4, 15.0), (4.4, 15.0), (4.4, 15.4), (-4.4, 15.4)]),
    ]
    corridor_paths = [
        PathSpec(points=[(-15.0, 0.0), (0.0, 0.0), (0.0, 15.0)]),
        PathSpec(points=[(0.0, 15.0), (0.0, 0.0), (-15.0, 0.0)]),
    ]

    scene = Scene(
            agents=[],
            obstacles=corridor_obstacles,
            paths=corridor_paths,
            region_pairs=[
                RegionPairSpec(
                    spawn_region=RegionSpec(min_corner=(-14.5, -2.0), max_corner=(-10.0, 2.0)),
                    destination_region=RegionSpec(min_corner=(-2.0, 13.0), max_corner=(2.0, 14.5)),
                    startup_agent_count=8,
                    path_index=0,
                ),
                RegionPairSpec(
                    spawn_region=RegionSpec(min_corner=(-2.0, 10.0), max_corner=(2.0, 14.5)),
                    destination_region=RegionSpec(min_corner=(-14.5, -2.0), max_corner=(-13.0, 2.0)),
                    startup_agent_count=8,
                    path_index=1,
                ),
            ],
        )
    return scene

def main() -> None:
    """Run an ORCA rollout with optional animation and occupancy-map generation."""
    parser = argparse.ArgumentParser(description="Run an ORCA pedestrian rollout.")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Display a matplotlib animation of the agents.",
    )
    args = parser.parse_args()
    template = StraightCorridorTemplate(
        width_range=(3.0, 6.0),
        length_range=(8.0, 15.0),
        startup_agent_count_per_pair=8,
        num_region_pairs=1,
    )
    scenes = template.generate(num_levels=4)
    print(f"generated {len(scenes)} scenes from StraightCorridorTemplate")

    for scene_index, scene in enumerate(scenes):
        orca_sim = ORCASim(scene=scene, time_step=args.dt, region_pair_seed=scene_index)
        traj = orca_sim.simulate(steps=args.steps, stop_on_goal=True)
        goals = np.array([agent.goal for agent in scene.agents], dtype=np.float32)
        occupancy_grids, occupancy_origin, occupancy_resolution = build_occupancy_maps(
            traj=traj,
            goals=goals,
            obstacles=scene.obstacles,
            resolution=OCC_RESOLUTION,
            margin=OCC_MARGIN,
            agent_radius=OCC_AGENT_RADIUS,
        )

        print(
            f"scene[{scene_index}] startup spawn: total agents={traj.shape[1]}, "
            f"steps={traj.shape[0]}"
        )

        if args.animate:
            animate_rollout(
                traj=traj,
                goals=goals,
                obstacles=scene.obstacles,
                occupancy_grids=occupancy_grids,
                occupancy_origin=occupancy_origin,
                occupancy_resolution=occupancy_resolution,
                time_step=args.dt,
                title_prefix=f"Scene {scene_index + 1}/{len(scenes)}",
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


if __name__ == "__main__":
    main()
