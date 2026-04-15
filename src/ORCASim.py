from __future__ import annotations

import warnings
from typing import List, Mapping

import numpy as np

try:
    import rvo2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "rvo2 is required for ORCA simulation. Install with `pip install git+https://github.com/sybrenstuvel/python-rvo2.git`."
    ) from exc


class ORCASim:
    """ORCA simulator wrapper with optional reference-path guided navigation."""

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
        region_pair_seed: int | None = 0,
        pref_velocity_noise_std: float = 0.0,
        pref_velocity_noise_interval: int = 1,
        pref_velocity_noise_seed: int | None = 0,
        lateral_control_gain: float = 1.0,
        lateral_control_max_speed: float | None = None,
        strict_controlled_agent_index: int | None = None,
        strict_control_velocity_tolerance: float = 1e-3,
        strict_control_assert: bool = False,
    ) -> None:
        """Initialize ORCA simulator and scene-dependent guidance parameters.

        Args:
            scene: Scene object containing agents, obstacles, and optional paths.
            time_step: Simulation step size.
            neighbor_dist: Neighbor search radius for ORCA.
            max_neighbors: Maximum neighbors considered per agent.
            time_horizon: Agent-agent avoidance time horizon.
            time_horizon_obst: Agent-obstacle avoidance time horizon.
            radius: Agent radius used by ORCA.
            max_speed: Agent speed upper bound.
            goal_tolerance: Distance threshold for considering goal reached.
            path_goal_switch_tolerance: Projection-gap threshold for switching to direct goal motion.
            path_segment_remaining_switch_ratio: Fraction of segment length remaining below which
                preferred direction switches to the next segment.
            region_pair_seed: Random seed used for startup region-pair spawning when
                the scene is configured with region pairs and no explicit agents.
            pref_velocity_noise_std: Standard deviation of Gaussian noise added to
                preferred velocity components to break symmetric deadlocks.
            pref_velocity_noise_interval: Number of simulation steps between noise
                resampling events (must be >= 1).
            pref_velocity_noise_seed: Random seed for preferred-velocity noise.
            lateral_control_gain: Proportional gain for path-lateral distance control.
            lateral_control_max_speed: Absolute cap for lateral correction velocity.
                If None, defaults to `0.5 * max_speed`.
            strict_controlled_agent_index: Optional single controlled scene-agent index.
                When set, this agent is configured to ignore ORCA avoidance and follow
                externally commanded preferred velocity each step.
            strict_control_velocity_tolerance: Allowed L2 velocity error between
                commanded and realized controlled-agent velocity.
            strict_control_assert: If True, raise when strict-control constraints cannot
                be enforced or velocity tracking error exceeds tolerance. If False,
                emit warnings.
        """
        if pref_velocity_noise_std < 0.0:
            raise ValueError("pref_velocity_noise_std must be >= 0")
        if pref_velocity_noise_interval < 1:
            raise ValueError("pref_velocity_noise_interval must be >= 1")
        if strict_control_velocity_tolerance < 0.0:
            raise ValueError("strict_control_velocity_tolerance must be >= 0")

        self.scene = scene
        self.time_step = time_step
        self.max_speed = float(max_speed)
        self.goal_tolerance = goal_tolerance
        self.path_goal_switch_tolerance = path_goal_switch_tolerance
        self.path_segment_remaining_switch_ratio = path_segment_remaining_switch_ratio
        self.pref_velocity_noise_std = float(pref_velocity_noise_std)
        self.pref_velocity_noise_interval = int(pref_velocity_noise_interval)
        self.strict_controlled_agent_index = (
            None
            if strict_controlled_agent_index is None
            else int(strict_controlled_agent_index)
        )
        self.strict_control_velocity_tolerance = float(strict_control_velocity_tolerance)
        self.strict_control_assert = bool(strict_control_assert)
        self._strict_control_configured = False
        self._strict_control_expected_velocity: np.ndarray | None = None
        self.lateral_control_gain = float(lateral_control_gain)
        if lateral_control_max_speed is None:
            self.lateral_control_max_speed = 0.5 * self.max_speed
        else:
            self.lateral_control_max_speed = float(max(0.0, lateral_control_max_speed))
        self._pref_velocity_rng = np.random.default_rng(pref_velocity_noise_seed)
        self._pref_velocity_step_count = 0
        self.sim = rvo2.PyRVOSimulator(
            time_step,
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            radius,
            max_speed,
        )
        self.agent_desired_speeds: List[float] = []
        self.agent_direct_goal_latched: List[bool] = []
        self.agent_pref_velocity_noise: List[np.ndarray] = []
        self.agent_reference_lateral_offsets: List[float | None] = []
        self._setup_obstacles()
        self.agent_ids = self._setup_agents()
        self._region_pairs_initialized = False
        if len(self.scene.agents) == 0 and len(getattr(self.scene, "region_pairs", [])) > 0:
            self.initialize_agents_from_region_pairs(seed=region_pair_seed)
        self._maybe_configure_strict_controlled_agent()

    def _maybe_configure_strict_controlled_agent(self) -> None:
        """Configure strict-control mode once when a controlled agent is requested."""
        if self.strict_controlled_agent_index is None or self._strict_control_configured:
            return

        idx = int(self.strict_controlled_agent_index)
        if idx < 0 or idx >= len(self.agent_ids):
            raise ValueError(
                f"strict_controlled_agent_index {idx} is out of range for "
                f"{len(self.agent_ids)} agents"
            )

        self._configure_strict_controlled_agent(idx)
        self._strict_control_configured = True

    def _configure_strict_controlled_agent(self, controlled_idx: int) -> None:
        """Disable ORCA avoidance for the controlled agent while keeping others reactive."""
        agent_id = self.agent_ids[controlled_idx]

        required_setters = (
            "setAgentMaxNeighbors",
            "setAgentNeighborDist",
            "setAgentTimeHorizon",
            "setAgentTimeHorizonObst",
        )
        missing = [name for name in required_setters if not hasattr(self.sim, name)]
        if missing:
            message = (
                "Strict control requested but python-rvo2 binding is missing required "
                f"per-agent setters: {missing}."
            )
            if self.strict_control_assert:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
            return

        self.sim.setAgentMaxNeighbors(agent_id, 0)
        self.sim.setAgentNeighborDist(agent_id, 0.0)
        self.sim.setAgentTimeHorizon(agent_id, 0.0)
        self.sim.setAgentTimeHorizonObst(agent_id, 0.0)

    def _set_controlled_agent_max_speed_if_needed(
        self,
        controlled_idx: int,
        velocity: np.ndarray,
    ) -> None:
        """Raise controlled-agent max speed to avoid internal clipping of commands."""
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_speed + 1e-9:
            return

        if not hasattr(self.sim, "setAgentMaxSpeed"):
            message = (
                "Controlled command speed exceeds simulator max_speed, but "
                "python-rvo2 binding lacks setAgentMaxSpeed; command may be clipped."
            )
            if self.strict_control_assert:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
            return

        agent_id = self.agent_ids[controlled_idx]
        self.sim.setAgentMaxSpeed(agent_id, speed + 1e-4)

    def _verify_strict_control_velocity_tracking(self, realized_velocities: np.ndarray) -> None:
        """Validate controlled-agent realized velocity matches commanded velocity."""
        idx = self.strict_controlled_agent_index
        expected = self._strict_control_expected_velocity
        if idx is None or expected is None:
            return

        realized = np.asarray(realized_velocities[idx], dtype=np.float32)
        error = float(np.linalg.norm(realized - expected))
        if error <= self.strict_control_velocity_tolerance:
            return

        message = (
            "Strict-control velocity mismatch for controlled agent "
            f"{idx}: error={error:.6f}, tolerance={self.strict_control_velocity_tolerance:.6f}, "
            f"expected={expected.tolist()}, realized={realized.tolist()}"
        )
        if self.strict_control_assert:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)

    def _setup_agents(self) -> List[int]:
        """Create ORCA agents from scene specifications and return simulator IDs."""
        ids: List[int] = []
        for idx, agent in enumerate(self.scene.agents):
            agent_id = self.sim.addAgent(agent.position)
            ids.append(agent_id)
            self.agent_desired_speeds.append(self.max_speed)
            self.agent_direct_goal_latched.append(False)
            self.agent_pref_velocity_noise.append(np.zeros(2, dtype=np.float32))
            self._initialize_agent_lateral_state(
                agent_index=idx,
                initial_position=np.array(agent.position, dtype=np.float32),
            )
        return ids

    def _compute_signed_lateral_distance_to_path(
        self,
        point: np.ndarray,
        path_points: list[tuple[float, float]],
    ) -> float | None:
        """Return signed distance from point to path using local path orientation."""
        closest, path_direction, _, _ = self._closest_point_and_direction_on_path(point, path_points)
        dir_norm = float(np.linalg.norm(path_direction))
        if dir_norm <= 1e-6:
            return None

        unit_dir = (path_direction / dir_norm).astype(np.float32)
        # Left-hand normal of path direction.
        normal = np.array([-unit_dir[1], unit_dir[0]], dtype=np.float32)
        signed_distance = float(np.dot(point - closest, normal))
        return signed_distance

    def _initialize_agent_lateral_state(
        self,
        agent_index: int,
        initial_position: np.ndarray,
    ) -> None:
        """Record initial path-relative lateral offset for proportional control."""
        path_index = getattr(self.scene.agents[agent_index], "path_index", None)
        scene_paths = getattr(self.scene, "paths", [])

        ref_offset: float | None = None
        if path_index is not None and 0 <= path_index < len(scene_paths):
            ref_offset = self._compute_signed_lateral_distance_to_path(
                initial_position,
                scene_paths[path_index].points,
            )

        self.agent_reference_lateral_offsets.append(ref_offset)

    def _compute_lateral_control_velocity(
        self,
        agent_index: int,
        pos: np.ndarray,
        preferred_direction: np.ndarray,
    ) -> np.ndarray:
        """Compute proportional lateral correction that keeps initial path-relative offset."""
        if self.lateral_control_max_speed <= 1e-9:
            return np.zeros(2, dtype=np.float32)

        path_index = getattr(self.scene.agents[agent_index], "path_index", None)
        scene_paths = getattr(self.scene, "paths", [])
        if path_index is None or path_index < 0 or path_index >= len(scene_paths):
            return np.zeros(2, dtype=np.float32)

        path_points = scene_paths[path_index].points
        current_signed_offset = self._compute_signed_lateral_distance_to_path(pos, path_points)
        if current_signed_offset is None:
            return np.zeros(2, dtype=np.float32)

        reference_offset = self.agent_reference_lateral_offsets[agent_index]
        if reference_offset is None:
            reference_offset = current_signed_offset
            self.agent_reference_lateral_offsets[agent_index] = reference_offset

        dir_norm = float(np.linalg.norm(preferred_direction))
        if dir_norm <= 1e-6:
            return np.zeros(2, dtype=np.float32)

        error = float(reference_offset - current_signed_offset)
        lateral_speed = self.lateral_control_gain * error
        lateral_speed = float(
            np.clip(
                lateral_speed,
                -self.lateral_control_max_speed,
                self.lateral_control_max_speed,
            )
        )

        unit_dir = (preferred_direction / dir_norm).astype(np.float32)
        # Lateral correction is perpendicular to preferred direction.
        lateral_unit = np.array([-unit_dir[1], unit_dir[0]], dtype=np.float32)
        return lateral_unit * lateral_speed

    @staticmethod
    def _sample_point_in_region(region, rng: np.random.Generator) -> tuple[float, float]:
        """Sample a random point uniformly inside an axis-aligned rectangular region."""
        min_corner = np.array(region.min_corner, dtype=np.float32)
        max_corner = np.array(region.max_corner, dtype=np.float32)
        low = np.minimum(min_corner, max_corner)
        high = np.maximum(min_corner, max_corner)
        sample = rng.uniform(low=low, high=high)
        return float(sample[0]), float(sample[1])

    def _sample_max_speed(
        self,
        velocity_range: tuple[float, float] | None,
        rng: np.random.Generator,
    ) -> float:
        """Sample agent max speed from an optional range; fall back to simulator default."""
        if velocity_range is None:
            return self.max_speed

        min_speed = float(velocity_range[0])
        max_speed = float(velocity_range[1])
        low = min(min_speed, max_speed)
        high = max(min_speed, max_speed)
        if low < 0.0:
            raise ValueError("RegionPairSpec.velocity_range must be non-negative")

        return float(rng.uniform(low=low, high=high))

    def _setup_obstacles(self) -> None:
        """Register polygon obstacles in ORCA and finalize obstacle processing."""
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
        """Project a point to a line segment.

        Returns:
            A tuple `(closest_point, t)` where `t` is the clamped segment parameter in [0, 1].
        """
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
        """Find closest point on a polyline and preferred local travel direction.

        If the closest projection is near the end of a segment (remaining ratio below
        `path_segment_remaining_switch_ratio`), the direction of the next segment is used.

        Returns:
            `(closest_point, direction, segment_index, t_on_segment)`.
        """
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

    def _compute_preferred_direction(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        path_index: int | None,
        agent_index: int,
    ) -> np.ndarray:
        """Compute preferred movement direction for one agent.

        Agents without path association use direct goal direction.
        Path-associated agents follow path direction, except when the agent and goal
        project to nearby points on the path, in which case direct goal direction is used.
        """
        to_goal = goal - pos
        goal_dist = float(np.linalg.norm(to_goal))
        if goal_dist <= self.goal_tolerance:
            return np.zeros(2, dtype=np.float32)

        scene_paths = getattr(self.scene, "paths", [])

        if path_index is None or path_index < 0 or path_index >= len(scene_paths):
            return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

        if self.agent_direct_goal_latched[agent_index]:
            return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

        path_points = scene_paths[path_index].points
        pos_closest, path_direction, _, _ = self._closest_point_and_direction_on_path(
            pos, path_points
        )
        goal_closest, _, _, _ = self._closest_point_and_direction_on_path(goal, path_points)

        projection_gap = float(np.linalg.norm(goal_closest - pos_closest))
        if projection_gap <= self.path_goal_switch_tolerance:
            self.agent_direct_goal_latched[agent_index] = True
            return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

        if float(np.linalg.norm(path_direction)) > 1e-6:
            return path_direction

        return (to_goal / max(goal_dist, 1e-6)).astype(np.float32)

    def _normalize_controlled_pref_velocities(
        self,
        controlled_pref_velocities: Mapping[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray]:
        """Validate and normalize external preferred-velocity overrides.

        Keys are scene-agent indices. Values are 2D preferred velocity vectors.
        """
        if controlled_pref_velocities is None:
            if self.strict_controlled_agent_index is not None:
                raise ValueError(
                    "strict_controlled_agent_index is set, so controlled_pref_velocities "
                    "must include that agent every step"
                )
            return {}

        normalized: dict[int, np.ndarray] = {}
        strict_idx = self.strict_controlled_agent_index
        for raw_idx, raw_velocity in controlled_pref_velocities.items():
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.agent_ids):
                raise ValueError(f"controlled agent index {idx} is out of range")
            if strict_idx is not None and idx != strict_idx:
                raise ValueError(
                    "strict_controlled_agent_index is set; only that agent may be externally controlled"
                )

            velocity = np.asarray(raw_velocity, dtype=np.float32)
            if velocity.shape != (2,):
                raise ValueError("controlled preferred velocity must have shape (2,)")
            if not np.isfinite(velocity).all():
                raise ValueError("controlled preferred velocity contains non-finite values")

            speed = float(np.linalg.norm(velocity))
            if strict_idx is None and speed > self.max_speed + 1e-9:
                velocity = velocity * (self.max_speed / max(speed, 1e-9))

            normalized[idx] = velocity

        if strict_idx is not None and strict_idx not in normalized:
            raise ValueError(
                "strict_controlled_agent_index is set, but controlled_pref_velocities "
                "did not include that agent"
            )

        return normalized

    def _set_preferred_velocities(
        self,
        controlled_pref_velocities: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        """Update preferred velocity vectors for all agents before each ORCA step."""
        self._maybe_configure_strict_controlled_agent()
        controlled = self._normalize_controlled_pref_velocities(controlled_pref_velocities)
        self._strict_control_expected_velocity = None

        should_resample_noise = (
            self.pref_velocity_noise_std > 0.0
            and self._pref_velocity_step_count % self.pref_velocity_noise_interval == 0
        )

        if should_resample_noise:
            for idx in range(len(self.agent_pref_velocity_noise)):
                sampled_noise = self._pref_velocity_rng.normal(
                    loc=0.0,
                    scale=self.pref_velocity_noise_std,
                    size=2,
                ).astype(np.float32)
                self.agent_pref_velocity_noise[idx] = sampled_noise

        for idx, agent_id in enumerate(self.agent_ids):
            controlled_velocity = controlled.get(idx)
            if controlled_velocity is None:
                pos = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)
                goal = np.array(self.scene.agents[idx].goal, dtype=np.float32)
                path_index = getattr(self.scene.agents[idx], "path_index", None)
                direction = self._compute_preferred_direction(pos, goal, path_index, idx)
                desired_speed = self.agent_desired_speeds[idx]
                velocity = direction * desired_speed
                velocity = velocity + self._compute_lateral_control_velocity(
                    agent_index=idx,
                    pos=pos,
                    preferred_direction=direction,
                )

                if float(np.linalg.norm(direction)) > 1e-6 and self.pref_velocity_noise_std > 0.0:
                    velocity = velocity + self.agent_pref_velocity_noise[idx]
            else:
                velocity = controlled_velocity
                if self.strict_controlled_agent_index is not None and idx == self.strict_controlled_agent_index:
                    self._set_controlled_agent_max_speed_if_needed(idx, velocity)
                    self._strict_control_expected_velocity = velocity.copy()

            self.sim.setAgentPrefVelocity(agent_id, (float(velocity[0]), float(velocity[1])))

        self._pref_velocity_step_count += 1

    def get_agent_positions(self) -> np.ndarray:
        """Return current simulator positions with shape (N, 2)."""
        positions = np.zeros((len(self.agent_ids), 2), dtype=np.float32)
        for idx, agent_id in enumerate(self.agent_ids):
            positions[idx] = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)
        return positions

    def get_agent_velocities(self) -> np.ndarray:
        """Return current simulator velocities with shape (N, 2)."""
        velocities = np.zeros((len(self.agent_ids), 2), dtype=np.float32)
        for idx, agent_id in enumerate(self.agent_ids):
            velocities[idx] = np.array(self.sim.getAgentVelocity(agent_id), dtype=np.float32)
        return velocities

    def all_agents_reached_goals(self) -> bool:
        """Return True if all agents are within goal_tolerance of their goals."""
        for idx, agent_id in enumerate(self.agent_ids):
            pos = np.array(self.sim.getAgentPosition(agent_id), dtype=np.float32)
            goal = np.array(self.scene.agents[idx].goal, dtype=np.float32)
            if np.linalg.norm(goal - pos) > self.goal_tolerance:
                return False
        return True

    def step(
        self,
        controlled_pref_velocities: Mapping[int, np.ndarray] | None = None,
        return_velocities: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Advance simulation by one step with optional controlled preferred velocities.

        Args:
            controlled_pref_velocities: Optional mapping from scene-agent index to
                preferred velocity (2,) for this step.
            return_velocities: When True, also return per-agent velocities.

        Returns:
            positions or (positions, velocities), each shape (N, 2).
        """
        self._set_preferred_velocities(controlled_pref_velocities=controlled_pref_velocities)
        self.sim.doStep()

        positions = self.get_agent_positions()
        velocities = self.get_agent_velocities()
        self._verify_strict_control_velocity_tracking(velocities)
        if not return_velocities:
            return positions
        return positions, velocities

    def _spawn_from_region_pair(self, pair, rng: np.random.Generator) -> int:
        """Spawn one startup agent for a region pair using sampled spawn and destination points."""
        from src.scene import AgentSpec

        spawn_pos = self._sample_point_in_region(pair.spawn_region, rng)
        dest_pos = self._sample_point_in_region(pair.destination_region, rng)
        self.scene.agents.append(
            AgentSpec(
                position=spawn_pos,
                goal=dest_pos,
                path_index=getattr(pair, "path_index", None),
            )
        )
        agent_id = self.sim.addAgent(spawn_pos)
        sampled_speed = self._sample_max_speed(getattr(pair, "velocity_range", None), rng)
        self.sim.setAgentMaxSpeed(agent_id, sampled_speed)
        self.agent_desired_speeds.append(sampled_speed)
        self.agent_direct_goal_latched.append(False)
        self.agent_pref_velocity_noise.append(np.zeros(2, dtype=np.float32))
        self._initialize_agent_lateral_state(
            agent_index=len(self.scene.agents) - 1,
            initial_position=np.array(spawn_pos, dtype=np.float32),
        )
        return agent_id

    def initialize_agents_from_region_pairs(self, seed: int | None = None) -> None:
        """Spawn startup agents from all configured region pairs.

        This method is intended for startup-only region-pair scenes so that the
        resulting agent set is fixed and compatible with `simulate()`.
        """
        if self._region_pairs_initialized:
            return

        region_pairs = getattr(self.scene, "region_pairs", [])
        if not region_pairs:
            return

        rng = np.random.default_rng(seed)
        for pair in region_pairs:
            startup_count = int(max(0, getattr(pair, "startup_agent_count", 0)))
            for _ in range(startup_count):
                agent_id = self._spawn_from_region_pair(pair, rng)
                self.agent_ids.append(agent_id)

        self._region_pairs_initialized = True
        self._maybe_configure_strict_controlled_agent()

    def simulate(
        self,
        steps: int,
        min_steps: int = 0,
        stop_on_goal: bool = False,
        return_velocities: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Run the simulation for up to `steps` steps.

        If `stop_on_goal` is True the simulation will stop early when all agents
        are within `goal_tolerance` of their goals, but never before `min_steps`
        have been executed. Returns a numpy array of shape (T, N, 2) where T is
        the number of executed steps (<= steps).

        When `return_velocities` is True, returns a tuple `(positions, velocities)`
        where `velocities` has the same shape `(T, N, 2)` and values are from
        ORCA's per-agent velocity state after each simulation step.
        """

        traj = np.zeros((steps, len(self.agent_ids), 2), dtype=np.float32)
        vel_traj = np.zeros((steps, len(self.agent_ids), 2), dtype=np.float32)
        for step in range(steps):
            step_positions, step_velocities = self.step(return_velocities=True)
            traj[step] = step_positions
            vel_traj[step] = step_velocities

            if stop_on_goal and (step + 1) >= min_steps:
                if self.all_agents_reached_goals():
                    if return_velocities:
                        return traj[: step + 1], vel_traj[: step + 1]
                    return traj[: step + 1]

        if return_velocities:
            return traj, vel_traj
        return traj


