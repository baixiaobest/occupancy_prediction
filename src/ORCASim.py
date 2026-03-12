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
    ) -> None:
        self.scene = scene
        self.time_step = time_step
        self.goal_tolerance = goal_tolerance
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

    def _set_preferred_velocities(self) -> None:
        for idx, agent_id in enumerate(self.agent_ids):
            pos = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)
            goal = np.array(self.scene.agents[idx].goal, dtype=np.float32)
            direction = goal - pos
            dist = np.linalg.norm(direction)
            # If agent is within goal_tolerance, stop (preferred velocity zero)
            if dist <= self.goal_tolerance:
                velocity = np.zeros(2, dtype=np.float32)
            elif dist > 1e-6:
                velocity = direction / dist
            else:
                velocity = np.zeros(2, dtype=np.float32)
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

