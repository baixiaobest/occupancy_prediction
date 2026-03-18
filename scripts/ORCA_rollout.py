from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import os
import sys
import torch

# Ensure project root is on sys.path so `from src...` works when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scene import ObstacleSpec, PathSpec
from src.ORCASim import ORCASim
from src.occupancy2d import Occupancy2d
from src.rollout_data import RollOutData
from src.rollout_setting import RollOutSetting
from src.templates import default_templates


# RollOutSetting is defined in src.rollout_setting


def build_occupancy_maps(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    ego_centers: List[Tuple[float, float]] | np.ndarray,
    resolution: float,
    margin: float,
    agent_radius: float,
    occupancy_length: float | None = None,
    occupancy_width: float | None = None,
) -> Tuple[List[List[torch.Tensor]], List[np.ndarray], Tuple[float, float]]:
    """Build per-timestep occupancy grids from rollout trajectory and static obstacles.

    Args:
        traj: Rollout trajectory with shape (T, N, 2).
        goals: Goal positions with shape (N, 2).
        obstacles: Static polygon obstacles.
        ego_centers: List of world-frame `(x, y)` centers used to build occupancy maps.
        resolution: Occupancy grid resolution in meters per cell.
        margin: Padding applied to automatic sizing bounds.
        agent_radius: Agent rasterization radius.
        occupancy_length: Optional map length along the x-axis. If omitted,
            x-size is derived from trajectory/goals/obstacles bounds plus margin.
        occupancy_width: Optional map width along the y-axis. If omitted,
            y-size is derived from trajectory/goals/obstacles bounds plus margin.

    Returns:
        A tuple `(all_grids, origins, resolution_xy)` where each entry in
        `all_grids` is a per-timestep occupancy sequence centered at the matching
        entry in `origins`.
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

    center = 0.5 * (min_xy + max_xy)
    size = auto_size.copy()

    if occupancy_length is not None:
        size[0] = float(occupancy_length)

    if occupancy_width is not None:
        size[1] = float(occupancy_width)

    centers_arr = np.asarray(ego_centers, dtype=np.float32)
    if centers_arr.size == 0:
        centers_arr = center.reshape(1, 2)
    else:
        if centers_arr.ndim != 2 or centers_arr.shape[1] != 2:
            raise ValueError("ego_centers must have shape (K, 2)")

    all_grids: List[List[torch.Tensor]] = []
    origins: List[np.ndarray] = []

    for center_xy in centers_arr:
        origin = np.array(
            [
                float(center_xy[0] - 0.5 * size[0]),
                float(center_xy[1] - 0.5 * size[1]),
            ],
            dtype=np.float32,
        )

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
        all_grids.append(occ2d.generate())
        origins.append(origin)

    return all_grids, origins, (resolution, resolution)


def data_augmentation(
    occupancy_grids: List[List[torch.Tensor]],
) -> Tuple[
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
    List[List[torch.Tensor]],
]:
    """Return split occupancy variants: original, mirror, rot90, rot180, rot270.

    Rotations are clockwise around the grid center.
    """

    original: List[List[torch.Tensor]] = []
    mirrored: List[List[torch.Tensor]] = []
    rot90: List[List[torch.Tensor]] = []
    rot180: List[List[torch.Tensor]] = []
    rot270: List[List[torch.Tensor]] = []

    for sequence in occupancy_grids:
        if not sequence:
            continue

        # Stack once so transforms are applied consistently to all timesteps.
        seq_tensor = torch.stack([torch.as_tensor(frame) for frame in sequence], dim=0)

        original.append([frame.clone() for frame in seq_tensor])
        mirrored.append([frame.clone() for frame in torch.flip(seq_tensor, dims=(-1,))])
        rot90.append([frame.clone() for frame in torch.rot90(seq_tensor, k=-1, dims=(-2, -1))])
        rot180.append([frame.clone() for frame in torch.rot90(seq_tensor, k=2, dims=(-2, -1))])
        rot270.append([frame.clone() for frame in torch.rot90(seq_tensor, k=-3, dims=(-2, -1))])

    return original, mirrored, rot90, rot180, rot270


def animate_rollout(
    traj: np.ndarray,
    goals: np.ndarray,
    obstacles: List[ObstacleSpec],
    paths: List[PathSpec],
    occupancy_grids: List[List[np.ndarray]],
    occupancy_origins: List[np.ndarray],
    occupancy_resolution: Tuple[float, float],
    time_step: float,
    title_prefix: str = "",
) -> None:
    """Animate trajectory and all occupancy maps in synchronized matplotlib windows."""
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

    def update(frame: int):
        scat.set_offsets(traj[frame])
        traj_time_text.set_text(f"t={frame * time_step:.2f}s")
        artists = [scat, traj_time_text]
        for occ_idx, (im, time_text) in enumerate(zip(occ_images, occ_time_texts)):
            im.set_data(occupancy_grids[occ_idx][frame])
            time_text.set_text(f"t={frame * time_step:.2f}s")
            artists.extend([im, time_text])
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
def main() -> None:
    """Run an ORCA rollout with optional animation and occupancy-map generation."""
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
    args = parser.parse_args()

    SAVE_ROLLOUTS = bool(args.save_rollouts)

    if args.data_dir is None:
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    else:
        DATA_DIR = os.path.abspath(os.path.expanduser(args.data_dir))

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
    OCC_MARGIN = 0.2
    OCC_AGENT_RADIUS = 0.2
    OCC_LENGTH = 12.8
    OCC_WIDTH = 12.8

    # premade templates are provided by `src.templates.default_templates()`
    rollout_setting = RollOutSetting(
        templates=default_templates(),
        mirror=False,
        rotate=False,
        name="default",
    )

    global_scene_index = 0
    for tpl in rollout_setting.templates:
        template_name = tpl.get_name()
        template_rollouts_original: List[RollOutData] = []
        template_rollouts_mirrored: List[RollOutData] = []
        template_rollouts_rot90: List[RollOutData] = []
        template_rollouts_rot180: List[RollOutData] = []
        template_rollouts_rot270: List[RollOutData] = []
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
            traj = orca_sim.simulate(steps=NUM_STEPS, stop_on_goal=True)
            goals = np.array([agent.goal for agent in scene.agents], dtype=np.float32)
            occupancy_grids, occupancy_origin, occupancy_resolution = build_occupancy_maps(
                traj=traj,
                goals=goals,
                obstacles=scene.obstacles,
                ego_centers=scene.ego_centers,
                resolution=OCC_RESOLUTION,
                margin=OCC_MARGIN,
                agent_radius=OCC_AGENT_RADIUS,
                occupancy_width=OCC_WIDTH,
                occupancy_length=OCC_LENGTH,
            )
            if DATA_AUG_ENABLED:
                (
                    occupancy_original,
                    occupancy_mirrored,
                    occupancy_rot90,
                    occupancy_rot180,
                    occupancy_rot270,
                ) = data_augmentation(occupancy_grids)
            else:
                occupancy_original = occupancy_grids
                occupancy_mirrored = []
                occupancy_rot90 = []
                occupancy_rot180 = []
                occupancy_rot270 = []

            if SAVE_ROLLOUTS:
                template_rollouts_original.append(
                    RollOutData(occupancy_grids=occupancy_original, dt=TIME_STEP)
                )
                if DATA_AUG_ENABLED:
                    template_rollouts_mirrored.append(
                        RollOutData(occupancy_grids=occupancy_mirrored, dt=TIME_STEP)
                    )
                    template_rollouts_rot90.append(
                        RollOutData(occupancy_grids=occupancy_rot90, dt=TIME_STEP)
                    )
                    template_rollouts_rot180.append(
                        RollOutData(occupancy_grids=occupancy_rot180, dt=TIME_STEP)
                    )
                    template_rollouts_rot270.append(
                        RollOutData(occupancy_grids=occupancy_rot270, dt=TIME_STEP)
                    )
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
                # Keep visualization focused on original maps.
                occupancy_grids_np = [
                [grid.detach().cpu().numpy() for grid in per_center_grids]
                for per_center_grids in occupancy_original
                ]

                animate_rollout(
                    traj=traj,
                    goals=goals,
                    obstacles=scene.obstacles,
                    paths=scene.paths,
                    occupancy_grids=occupancy_grids_np,
                    occupancy_origins=occupancy_origin,
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
                    f"scene[{scene_index}] occupancy generated: "
                    f"orig={len(occupancy_original)}, mirror={len(occupancy_mirrored)}, "
                    f"rot90={len(occupancy_rot90)}, rot180={len(occupancy_rot180)}, "
                    f"rot270={len(occupancy_rot270)} maps; "
                    f"{len(occupancy_original[0])} frames each, "
                    f"grid shape {occupancy_original[0][0].shape}"
                )

            global_scene_index += 1

        if SAVE_ROLLOUTS:
            outputs = [(f"rollout_{template_name}_orig.pt", template_rollouts_original)]
            if DATA_AUG_ENABLED:
                outputs.extend(
                    [
                        (f"rollout_{template_name}_mirror.pt", template_rollouts_mirrored),
                        (f"rollout_{template_name}_rot90.pt", template_rollouts_rot90),
                        (f"rollout_{template_name}_rot180.pt", template_rollouts_rot180),
                        (f"rollout_{template_name}_rot270.pt", template_rollouts_rot270),
                    ]
                )
            for file_name, payload in outputs:
                data_path = os.path.join(DATA_DIR, file_name)
                torch.save(payload, data_path)
                print(
                    f"saved template rollout data: {data_path} "
                    f"({len(payload)} scenes)"
                )


if __name__ == "__main__":
    main()
