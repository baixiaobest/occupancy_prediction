from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure project root is on sys.path so `from src...` works when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ORCASim import ORCASim
from src.rollout_data import SceneRollOutData
from src.rollout_helpers import (
    append_scene_rollout_to_template,
    build_agent_centric_occupancy_sequences,
    build_local_windows_over_time,
    print_scene_occupancy_summary,
    save_scene_rollouts,
    save_template_rollouts,
)
from src.rollout_setting import RollOutSetting
from src.rollout_visualization import (
    animate_rollout,
    prepare_animation_grids,
    prepare_past_future_dynamic_grids,
)
from src.templates import cross_templates, default_templates, test_templates


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ORCA pedestrian rollout.")
    parser.add_argument(
        "--template-set",
        type=str,
        choices=["default", "test", "cross"],
        default="cross",
        help="Template function to use: default_templates, test_templates, or cross_templates.",
    )
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
        "--occ-past-frames",
        type=int,
        default=16,
        help="Past frame count rendered on each occupancy grid.",
    )
    parser.add_argument(
        "--occ-future-frames",
        type=int,
        default=16,
        help="Future frame count rendered on each occupancy grid.",
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


def _select_templates(template_set: str):
    if template_set == "default":
        return default_templates(), "default"
    if template_set == "test":
        return test_templates(), "test"
    if template_set == "cross":
        return cross_templates(), "cross"
    raise ValueError(f"Unknown template set: {template_set}")


def main() -> None:
    """Run an ORCA rollout with optional animation and occupancy-map generation."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    save_rollouts = bool(args.save_rollouts)
    data_dir = _resolve_data_dir(args.data_dir)
    if save_rollouts:
        os.makedirs(data_dir, exist_ok=True)

    animate = bool(args.animate)

    # ORCASim configuration constants
    time_step = 0.1
    num_steps = 200
    neighbor_dist = 1.0
    max_neighbors = 5
    time_horizon = 3.0
    time_horizon_obst = 5.0
    radius = 0.3
    max_speed = 3.0
    goal_tolerance = 0.2
    path_goal_switch_tolerance = 3.0
    path_segment_remaining_switch_ratio = 0.05
    pref_velocity_noise_std = 0.02
    pref_velocity_noise_interval = 3
    pref_velocity_noise_seed = 0
    lateral_control_gain = 1.0
    lateral_control_max_speed = 1.0

    # Occupancy settings
    occ_resolution = 0.1
    occ_agent_radius = 0.2
    occ_length = 12.8
    occ_width = 12.8

    selected_templates, selected_template_set = _select_templates(args.template_set)
    rollout_setting = RollOutSetting(
        templates=selected_templates,
        mirror=False,
        rotate=False,
        name=selected_template_set,
    )

    print(f"using template set: {selected_template_set}")

    global_scene_index = 0
    for tpl in rollout_setting.templates:
        template_name = tpl.get_name()
        template_rollouts: list[SceneRollOutData] = []
        scenes = tpl.generate()
        print(
            f"generated {len(scenes)} scenes from {tpl.__class__.__name__} "
            f"(name={template_name}, num_levels={int(tpl.num_levels)})"
        )

        for local_idx, scene in enumerate(scenes):
            scene_index = global_scene_index
            orca_sim = ORCASim(
                scene=scene,
                time_step=time_step,
                neighbor_dist=neighbor_dist,
                max_neighbors=max_neighbors,
                time_horizon=time_horizon,
                time_horizon_obst=time_horizon_obst,
                radius=radius,
                max_speed=max_speed,
                goal_tolerance=goal_tolerance,
                path_goal_switch_tolerance=path_goal_switch_tolerance,
                path_segment_remaining_switch_ratio=path_segment_remaining_switch_ratio,
                region_pair_seed=scene_index,
                pref_velocity_noise_std=pref_velocity_noise_std,
                pref_velocity_noise_interval=pref_velocity_noise_interval,
                pref_velocity_noise_seed=pref_velocity_noise_seed + scene_index,
                lateral_control_gain=lateral_control_gain,
                lateral_control_max_speed=lateral_control_max_speed,
            )
            min_steps_for_occupancy = int(args.occ_past_frames) + int(args.occ_future_frames) + 1
            traj, vel_traj = orca_sim.simulate(
                steps=num_steps,
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
                position_trajectories,
                velocity_trajectories,
                scene_map_origin,
                occupancy_resolution,
                frame_offsets,
                local_map_shape,
            ) = build_agent_centric_occupancy_sequences(
                traj=traj,
                velocities=vel_traj,
                obstacles=scene.obstacles,
                resolution=occ_resolution,
                agent_radius=occ_agent_radius,
                occupancy_width=occ_width,
                occupancy_length=occ_length,
                past_frames=int(args.occ_past_frames),
                future_frames=int(args.occ_future_frames),
            )

            if save_rollouts:
                if args.save_every_scene:
                    save_scene_rollouts(
                        data_dir=data_dir,
                        template_name=template_name,
                        scene_index=scene_index,
                        dt=time_step,
                        occupancy_resolution=occupancy_resolution,
                        occupancy_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
                        frame_offsets=frame_offsets,
                        total_steps=int(traj.shape[0]),
                        scene_static_map=scene_static_map,
                        dynamic_maps=dynamic_maps,
                        position_trajectories=position_trajectories,
                        velocity_trajectories=velocity_trajectories,
                        scene_map_origin=scene_map_origin,
                        local_map_shape=local_map_shape,
                    )
                else:
                    append_scene_rollout_to_template(
                        template_rollouts=template_rollouts,
                        dt=time_step,
                        occupancy_resolution=occupancy_resolution,
                        occupancy_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
                        frame_offsets=frame_offsets,
                        total_steps=int(traj.shape[0]),
                        scene_static_map=scene_static_map,
                        dynamic_maps=dynamic_maps,
                        position_trajectories=position_trajectories,
                        velocity_trajectories=velocity_trajectories,
                        scene_map_origin=scene_map_origin,
                        local_map_shape=local_map_shape,
                    )
                    print(
                        f"scene[{scene_index}] queued rollout data for template "
                        f"{template_name} (orig only)"
                    )

            print(
                f"scene[{scene_index}] startup spawn: total agents={traj.shape[1]}, "
                f"steps={traj.shape[0]}"
            )

            title_prefix = f"{tpl.__class__.__name__} {local_idx + 1}/{len(scenes)}"
            if animate:
                static_maps, dynamic_windows = build_local_windows_over_time(
                    scene_static_map=scene_static_map,
                    dynamic_maps=dynamic_maps,
                    center_trajectories=position_trajectories,
                    frame_offsets=frame_offsets,
                    scene_origin=scene_map_origin,
                    occupancy_resolution=occupancy_resolution,
                    local_map_shape=local_map_shape,
                    total_steps=int(traj.shape[0]),
                )
                static_maps_np = prepare_animation_grids(static_maps)
                dynamic_past_grids_np, dynamic_future_grids_np = prepare_past_future_dynamic_grids(
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
                    velocities=vel_traj,
                    goals=goals,
                    obstacles=scene.obstacles,
                    paths=scene.paths,
                    static_maps=static_maps_np,
                    dynamic_past_grids=dynamic_past_grids_np,
                    dynamic_future_grids=dynamic_future_grids_np,
                    occupancy_origins=local_origins,
                    occupancy_resolution=occupancy_resolution,
                    time_step=time_step,
                    title_prefix=title_prefix,
                )
            else:
                print_scene_occupancy_summary(
                    scene_index=scene_index,
                    traj=traj,
                    scene_static_map=scene_static_map,
                    dynamic_maps=dynamic_maps,
                    velocity_trajectories=velocity_trajectories,
                    frame_offsets=frame_offsets,
                )

            global_scene_index += 1

        if save_rollouts and (not args.save_every_scene) and len(template_rollouts) > 0:
            save_template_rollouts(
                data_dir=data_dir,
                template_name=template_name,
                template_rollouts=template_rollouts,
            )


if __name__ == "__main__":
    main()
