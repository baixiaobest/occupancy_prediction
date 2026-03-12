from __future__ import annotations

from typing import List

import numpy as np

try:
    import rvo2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "rvo2 is required for ORCA simulation. Install with `pip install git+https://github.com/sybrenstuvel/python-rvo2.git`."
    ) from exc


class ORCASim:
    def __init__(
        self,
        scene,
        time_step: float = 0.1,
        neighbor_dist: float = 5.0,
        max_neighbors: int = 10,
        time_horizon: float = 5.0,
        time_horizon_obst: float = 5.0,
        radius: float = 0.3,
        max_speed: float = 1.5,
        goal_tolerance: float = 0.1,
        path_goal_switch_tolerance: float = 0.5,
        path_segment_remaining_switch_ratio: float = 0.1,
    ) -> None:
        self.scene = scene
        self.time_step = time_step
        self.goal_tolerance = goal_tolerance
        self.path_goal_switch_tolerance = path_goal_switch_tolerance
        self.path_segment_remaining_switch_ratio = path_segment_remaining_switch_ratio
        self.sim = rvo2.PyRVOSimulator(
            time_step,
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            radius,
            max_speed,
        )
        self._setup_obstacles()
        self.agent_ids = self._setup_agents()

    def _setup_agents(self) -> List[int]:
        ids: List[int] = []
        for agent in self.scene.agents:
            agent_id = self.sim.addAgent(agent.position)
            ids.append(agent_id)
        return ids

    def _setup_obstacles(self) -> None:
        for obstacle in self.scene.obstacles:
            if len(obstacle.vertices) < 2:
                continue
            self.sim.addObstacle(obstacle.vertices)
        if self.scene.obstacles:
            self.sim.processObstacles()

    @staticmethod
    def _closest_point_on_segment(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        segment = seg_end - seg_start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1e-12:
            return seg_start, 0.0

        t = float(np.dot(point - seg_start, segment) / seg_len_sq)
        t_clamped = float(np.clip(t, 0.0, 1.0))
        closest = seg_start + t_clamped * segment
        return closest, t_clamped

    def _closest_point_and_direction_on_path(
        self,
        point: np.ndarray,
        path_points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        if len(path_points) < 2:
            return point.copy(), np.zeros(2, dtype=np.float32), -1, 0.0

        best_dist_sq = float("inf")
        best_point = point.copy()
        best_dir = np.zeros(2, dtype=np.float32)
        best_seg_idx = -1
        best_t = 0.0

        for seg_idx in range(len(path_points) - 1):
            start = np.array(path_points[seg_idx], dtype=np.float32)
            end = np.array(path_points[seg_idx + 1], dtype=np.float32)
            segment = end - start
            seg_norm = float(np.linalg.norm(segment))
            if seg_norm <= 1e-6:
                continue

            closest, t_clamped = self._closest_point_on_segment(point, start, end)
            diff = point - closest
            dist_sq = float(np.dot(diff, diff))

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_point = closest
                best_dir = (segment / seg_norm).astype(np.float32)
                best_seg_idx = seg_idx
                best_t = t_clamped

        if best_seg_idx >= 0:
            remaining_ratio = 1.0 - best_t
            if remaining_ratio <= self.path_segment_remaining_switch_ratio:
                next_idx = best_seg_idx + 1
                if next_idx < len(path_points) - 1:
                    next_start = np.array(path_points[next_idx], dtype=np.float32)
                    next_end = np.array(path_points[next_idx + 1], dtype=np.float32)
                    next_segment = next_end - next_start
                    next_norm = float(np.linalg.norm(next_segment))
                    if next_norm > 1e-6:
                        best_dir = (next_segment / next_norm).astype(np.float32)

        return best_point.astype(np.float32), best_dir, best_seg_idx, best_t

    def _compute_preferred_direction(self, idx: int, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        to_goal = goal - pos
        goal_dist = float(np.linalg.norm(to_goal))
        if goal_dist <= self.goal_tolerance:
            return np.zeros(2, dtype=np.float32)

        agent_spec = self.scene.agents[idx]
        path_index = getattr(agent_spec, "path_index", None)
        scene_paths = getattr(self.scene, "paths", [])

        if path_index is None or path_index < 0 or path_index >= len(scene_paths):
            return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

        path_points = scene_paths[path_index].points
        pos_closest, path_direction, _, _ = self._closest_point_and_direction_on_path(
            pos, path_points
        )
        goal_closest, _, _, _ = self._closest_point_and_direction_on_path(goal, path_points)

        projection_gap = float(np.linalg.norm(goal_closest - pos_closest))
        if projection_gap <= self.path_goal_switch_tolerance:
            return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

        if float(np.linalg.norm(path_direction)) > 1e-6:
            return path_direction

        return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

    def _set_preferred_velocities(self) -> None:
        for idx, agent_id in enumerate(self.agent_ids):
            pos = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)
            goal = np.array(self.scene.agents[idx].goal, dtype=np.float32)
            velocity = self._compute_preferred_direction(idx, pos, goal)
            self.sim.setAgentPrefVelocity(agent_id, (float(velocity[0]), float(velocity[1])))

    def simulate(self, steps: int, stop_on_goal: bool = False) -> np.ndarray:
        """Run the simulation for up to `steps` steps.

        If `stop_on_goal` is True the simulation will stop early when all agents
        are within `goal_tolerance` of their goals. Returns a numpy array of
        shape (T, N, 2) where T is the number of executed steps (<= steps).
        """
        traj = np.zeros((steps, len(self.agent_ids), 2), dtype=np.float32)
        for step in range(steps):
            self._set_preferred_velocities()
            self.sim.doStep()
            for j, agent_id in enumerate(self.agent_ids):
                traj[step, j] = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)

            if stop_on_goal:
                # Check whether all agents are within tolerance to their goals
                all_reached = True
                for idx in range(len(self.agent_ids)):
                    pos = traj[step, idx]
                    goal = np.array(self.scene.agents[idx].goal, dtype=np.float32)
                    if np.linalg.norm(goal - pos) > self.goal_tolerance:
                        all_reached = False
                        break
                if all_reached:
                    return traj[: step + 1]

        return traj

