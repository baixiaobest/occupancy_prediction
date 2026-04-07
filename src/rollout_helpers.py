from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.occupancy2d import Occupancy2d
from src.occupancy_patch import slice_centered_patch
from src.rollout_data import AgentRollOutData, RollOutData, SceneRollOutData
from src.scene import ObstacleSpec


def _compute_scene_canvas(
    traj: np.ndarray,
    obstacles: List[ObstacleSpec],
    resolution: float,
    occupancy_length: float,
    occupancy_width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], np.ndarray]:
    """Compute global scene canvas origin/size/grid shape/center."""
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
    """Rasterize global dynamic occupancy over all timesteps for one centered agent."""
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


def _collect_anchor_centers(
    traj: np.ndarray,
    center_agent_idx: int,
    anchor_steps: List[int],
) -> torch.Tensor:
    """Collect anchor center positions for one agent."""
    center_series = torch.zeros((len(anchor_steps), 2), dtype=torch.float32)

    for anchor_idx, anchor_t in enumerate(anchor_steps):
        center_series[anchor_idx] = torch.as_tensor(
            traj[anchor_t, center_agent_idx], dtype=torch.float32
        )

    return center_series


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
    """Build scene-global occupancy maps and per-agent anchor metadata."""
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
    velocity_trajectories: List[torch.Tensor] = []
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
        center_series = _collect_anchor_centers(
            traj=traj,
            center_agent_idx=center_agent_idx,
            anchor_steps=anchor_steps,
        )

        dynamic_maps.append(agent_dynamic)
        velocity_trajectories.append(
            torch.as_tensor(velocities[:, center_agent_idx], dtype=torch.float32).clone()
        )
        anchor_centers.append(center_series)

    return (
        dynamic_maps,
        scene_static_map,
        velocity_trajectories,
        scene_origin,
        (resolution, resolution),
        anchor_steps,
        frame_offsets,
        anchor_centers,
        (local_h, local_w),
    )


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
            static_local = slice_centered_patch(
                scene_static_map,
                center_xy,
                scene_origin,
                occupancy_resolution,
                local_map_shape,
                binary=False,
                prefer_view=False,
            )
            agent_static_series.append(static_local)

            anchor_frames: List[torch.Tensor] = []
            for dt_offset in frame_offsets:
                absolute_t = int(anchor_t + dt_offset)
                if absolute_t < 0 or absolute_t >= int(total_steps):
                    anchor_frames.append(torch.zeros(local_map_shape, dtype=torch.float32))
                    continue
                dynamic_local = slice_centered_patch(
                    agent_dynamic[absolute_t],
                    center_xy,
                    scene_origin,
                    occupancy_resolution,
                    local_map_shape,
                    binary=False,
                    prefer_view=False,
                )
                anchor_frames.append(dynamic_local)

            agent_dynamic_windows.append(anchor_frames)

        static_maps.append(agent_static_series)
        dynamic_windows.append(agent_dynamic_windows)

    return static_maps, dynamic_windows


def build_scene_rollout_data(
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    velocity_trajectories: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> SceneRollOutData:
    if not (
        len(dynamic_maps)
        == len(velocity_trajectories)
        == len(anchor_centers)
    ):
        raise ValueError(
            "dynamic_maps, velocity_trajectories, anchor_centers must have same length"
        )

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

    scene_velocity_trajectories = torch.stack(
        [torch.as_tensor(v, dtype=torch.float32) for v in velocity_trajectories],
        dim=0,
    )
    if scene_velocity_trajectories.shape != (scene_dynamic_maps.shape[0], int(total_steps), 2):
        raise ValueError("velocity_trajectories must stack into shape (num_agents, total_steps, 2)")

    agents: Dict[int, AgentRollOutData] = {}

    for agent_idx, center_series in enumerate(anchor_centers):
        centers_tensor = torch.as_tensor(center_series, dtype=torch.float32)
        if centers_tensor.shape != (len(anchor_steps), 2):
            raise ValueError("center series must have shape (num_anchors, 2)")

        agents[agent_idx] = AgentRollOutData(
            agent_index=agent_idx,
            anchor_times=[int(t) for t in anchor_steps],
            anchor_centers=centers_tensor.clone(),
        )

    return SceneRollOutData(
        dt=float(dt),
        occupancy_resolution=(float(occupancy_resolution[0]), float(occupancy_resolution[1])),
        occupancy_origin=(float(occupancy_origin[0]), float(occupancy_origin[1])),
        frame_offsets=[int(v) for v in frame_offsets],
        agents=agents,
        scene_static_map=static_2d,
        scene_dynamic_maps=scene_dynamic_maps,
        scene_velocity_trajectories=scene_velocity_trajectories,
        scene_map_origin=(float(scene_map_origin[0]), float(scene_map_origin[1])),
        local_map_shape=(int(local_map_shape[0]), int(local_map_shape[1])),
    )


def _build_scene_rollout_payload(
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    velocity_trajectories: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> SceneRollOutData:
    return build_scene_rollout_data(
        dt=dt,
        occupancy_resolution=occupancy_resolution,
        occupancy_origin=occupancy_origin,
        frame_offsets=frame_offsets,
        anchor_steps=anchor_steps,
        total_steps=total_steps,
        scene_static_map=scene_static_map,
        dynamic_maps=dynamic_maps,
        velocity_trajectories=velocity_trajectories,
        anchor_centers=anchor_centers,
        scene_map_origin=scene_map_origin,
        local_map_shape=local_map_shape,
    )


def append_scene_rollout_to_template(
    template_rollouts: List[SceneRollOutData],
    *,
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    velocity_trajectories: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> None:
    rollout = _build_scene_rollout_payload(
        dt=dt,
        occupancy_resolution=occupancy_resolution,
        occupancy_origin=occupancy_origin,
        frame_offsets=frame_offsets,
        anchor_steps=anchor_steps,
        total_steps=total_steps,
        scene_static_map=scene_static_map,
        dynamic_maps=dynamic_maps,
        velocity_trajectories=velocity_trajectories,
        anchor_centers=anchor_centers,
        scene_map_origin=scene_map_origin,
        local_map_shape=local_map_shape,
    )
    template_rollouts.append(rollout)


def save_template_rollouts(
    *,
    data_dir: str,
    template_name: str,
    template_rollouts: List[SceneRollOutData],
) -> None:
    file_name = f"rollout_{template_name}_orig.pt"
    payload = RollOutData(scenes=template_rollouts)
    data_path = os.path.join(data_dir, file_name)
    torch.save(payload, data_path)
    print(f"saved template rollout data: {data_path} ({len(payload.scenes)} scenes)")


def save_scene_rollouts(
    *,
    data_dir: str,
    template_name: str,
    scene_index: int,
    dt: float,
    occupancy_resolution: Tuple[float, float],
    occupancy_origin: Tuple[float, float],
    frame_offsets: List[int],
    anchor_steps: List[int],
    total_steps: int,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    velocity_trajectories: List[torch.Tensor],
    anchor_centers: List[torch.Tensor],
    scene_map_origin: Tuple[float, float],
    local_map_shape: Tuple[int, int],
) -> None:
    scene_payload = _build_scene_rollout_payload(
        dt=dt,
        occupancy_resolution=occupancy_resolution,
        occupancy_origin=occupancy_origin,
        frame_offsets=frame_offsets,
        anchor_steps=anchor_steps,
        total_steps=total_steps,
        scene_static_map=scene_static_map,
        dynamic_maps=dynamic_maps,
        velocity_trajectories=velocity_trajectories,
        anchor_centers=anchor_centers,
        scene_map_origin=scene_map_origin,
        local_map_shape=local_map_shape,
    )
    payload = RollOutData(scenes=[scene_payload])
    file_name = f"rollout_{template_name}_scene{scene_index:05d}_orig.pt"
    data_path = os.path.join(data_dir, file_name)
    torch.save(payload, data_path)
    print(f"saved scene rollout: {data_path}")


def print_scene_occupancy_summary(
    scene_index: int,
    traj: np.ndarray,
    scene_static_map: torch.Tensor,
    dynamic_maps: List[torch.Tensor],
    velocity_trajectories: List[torch.Tensor],
    anchor_steps: List[int],
) -> None:
    for i, pos in enumerate(traj[-1]):
        print(
            f"scene[{scene_index}] agent[{i}] final position: "
            f"({pos[0]:.2f}, {pos[1]:.2f})"
        )

    dynamic_shape = tuple(dynamic_maps[0].shape) if dynamic_maps else ()
    trajectory_shape = tuple(velocity_trajectories[0].shape) if velocity_trajectories else ()
    print(
        f"scene[{scene_index}] occupancy generated: "
        f"orig_maps={len(dynamic_maps)}; "
        f"{len(anchor_steps)} anchors each (sampled), "
        f"global static shape {tuple(scene_static_map.shape)}, "
        f"global dynamic shape {dynamic_shape}, "
        f"velocity trajectory shape {trajectory_shape}"
    )
