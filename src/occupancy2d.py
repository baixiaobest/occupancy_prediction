from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class Occupancy2d:
	"""2D occupancy grid generator from ORCA rollout output.

	- Dynamic occupancy comes from agent positions in `trajectory` (shape: T x N x 2).
	- Static occupancy comes from polygon obstacles (e.g., Scene.obstacles).
	"""

	def __init__(
		self,
		resolution: Tuple[float, float] | torch.Tensor,
		size: Tuple[float, float] | torch.Tensor,
		trajectory: np.ndarray | torch.Tensor | None = None,
		static_obstacles: Sequence[object] | None = None,
		agent_radius: float = 0.3,
	) -> None:
		self.resolution = _to_tensor(resolution)
		self.size = _to_tensor(size)
		self.agent_radius = float(agent_radius)
		self.trajectory = _to_float_tensor(trajectory) if trajectory is not None else None
		self.static_obstacles: List[List[Tuple[float, float]]] = (
			_normalize_obstacles(static_obstacles) if static_obstacles is not None else []
		)

		if torch.any(self.resolution <= 0):
			raise ValueError("resolution must be positive")
		if torch.any(self.size <= 0):
			raise ValueError("size must be positive")
		if self.agent_radius < 0:
			raise ValueError("agent_radius must be non-negative")

	def update_inputs(
		self,
		trajectory: np.ndarray | torch.Tensor | None,
		static_obstacles: Sequence[object] | None = None,
	) -> None:
		self.trajectory = _to_float_tensor(trajectory) if trajectory is not None else None
		if static_obstacles is not None:
			self.static_obstacles = _normalize_obstacles(static_obstacles)

	def generate(
		self,
		center_offset: Tuple[float, float] | torch.Tensor | None = None,
	) -> List[torch.Tensor]:
		"""Generate a list of occupancy grids (one per timestep).

		Args:
			center_offset: optional absolute grid center (x, y) in world coordinates.
				When omitted, center defaults to (0.0, 0.0).

		Returns:
			List[torch.Tensor]: each grid has shape (H, W), with 1 occupied and 0 free.
		"""
		cells_x = int(torch.floor(self.size[0] / self.resolution[0]).item())
		cells_y = int(torch.floor(self.size[1] / self.resolution[1]).item())

		if cells_x <= 0 or cells_y <= 0:
			raise ValueError("size/resolution yields non-positive grid dimensions")

		if self.trajectory is not None:
			if self.trajectory.ndim != 3 or self.trajectory.shape[-1] != 2:
				raise ValueError("trajectory must have shape (T, N, 2)")
			trajectory = self.trajectory.to(dtype=torch.float32)
			device = trajectory.device
			num_steps = int(trajectory.shape[0])
			num_agents = int(trajectory.shape[1])
		else:
			trajectory = None
			device = self.resolution.device
			num_steps = 1
			num_agents = 0

		if center_offset is not None:
			offset_tensor = _to_tensor(center_offset, device=device).to(dtype=torch.float32)
			if offset_tensor.numel() != 2:
				raise ValueError("center_offset must have exactly two elements")
			grid_center = offset_tensor
		else:
			grid_center = torch.zeros(2, device=device, dtype=torch.float32)

		grids: List[torch.Tensor] = []
		for t in range(num_steps):
			grid = torch.zeros((cells_y, cells_x), dtype=torch.uint8, device=device)

			# Static obstacles are present in every frame.
			for polygon in self.static_obstacles:
				self._rasterize_polygon(grid, polygon, grid_center)

			# Dynamic occupancy from agent positions at this timestep.
			for n in range(num_agents):
				self._rasterize_agent(grid, trajectory[t, n], grid_center)

			grids.append(grid)

		return grids

	def _rasterize_agent(
		self,
		grid: torch.Tensor,
		position: torch.Tensor,
		grid_center: torch.Tensor,
	) -> None:
		half_size = torch.tensor(
			[self.agent_radius, self.agent_radius], device=grid.device, dtype=torch.float32
		)
		self._rasterize_box(grid, position.to(grid.device), half_size, grid_center)

	def _rasterize_box(
		self,
		grid: torch.Tensor,
		center_xy: torch.Tensor,
		half_size_xy: torch.Tensor,
		grid_center: torch.Tensor,
	) -> None:
		cells_y, cells_x = grid.shape
		resolution = self.resolution.to(device=grid.device, dtype=torch.float32)
		size = self.size.to(device=grid.device, dtype=torch.float32)
		grid_center = grid_center.to(device=grid.device, dtype=torch.float32)
		center_xy = center_xy.to(device=grid.device, dtype=torch.float32)
		half_size_xy = half_size_xy.to(device=grid.device, dtype=torch.float32)
		grid_min = grid_center - size * 0.5
		grid_max = grid_min + torch.tensor(
			[cells_x, cells_y], device=grid.device, dtype=torch.float32
		) * resolution

		min_xy = center_xy - half_size_xy
		max_xy = center_xy + half_size_xy

		if (
			max_xy[0] <= grid_min[0]
			or min_xy[0] >= grid_max[0]
			or max_xy[1] <= grid_min[1]
			or min_xy[1] >= grid_max[1]
		):
			return

		min_idx = torch.floor((min_xy - grid_min) / resolution).to(dtype=torch.long)
		max_idx = torch.floor((max_xy - grid_min) / resolution).to(dtype=torch.long)

		min_ix = int(torch.clamp(min_idx[0], 0, cells_x - 1).item())
		max_ix = int(torch.clamp(max_idx[0], 0, cells_x - 1).item())
		min_iy = int(torch.clamp(min_idx[1], 0, cells_y - 1).item())
		max_iy = int(torch.clamp(max_idx[1], 0, cells_y - 1).item())

		if min_ix > max_ix or min_iy > max_iy:
			return

		grid[min_iy : max_iy + 1, min_ix : max_ix + 1] = 1

	def _rasterize_polygon(
		self,
		grid: torch.Tensor,
		polygon: List[Tuple[float, float]],
		grid_center: torch.Tensor,
	) -> None:
		if len(polygon) < 3:
			return

		cells_y, cells_x = grid.shape
		polygon_np = np.asarray(polygon, dtype=np.float32)
		resolution = self.resolution.to(device=grid.device, dtype=torch.float32)
		size = self.size.to(device=grid.device, dtype=torch.float32)
		grid_center = grid_center.to(device=grid.device, dtype=torch.float32)

		min_x = float(np.min(polygon_np[:, 0]))
		max_x = float(np.max(polygon_np[:, 0]))
		min_y = float(np.min(polygon_np[:, 1]))
		max_y = float(np.max(polygon_np[:, 1]))

		res_x = float(resolution[0].item())
		res_y = float(resolution[1].item())
		grid_min_x = float((grid_center[0] - size[0] * 0.5).item())
		grid_min_y = float((grid_center[1] - size[1] * 0.5).item())

		min_ix = max(0, int(np.floor((min_x - grid_min_x) / res_x)))
		max_ix = min(cells_x - 1, int(np.floor((max_x - grid_min_x) / res_x)))
		min_iy = max(0, int(np.floor((min_y - grid_min_y) / res_y)))
		max_iy = min(cells_y - 1, int(np.floor((max_y - grid_min_y) / res_y)))

		if min_ix > max_ix or min_iy > max_iy:
			return

		for iy in range(min_iy, max_iy + 1):
			center_y = grid_min_y + (iy + 0.5) * res_y
			for ix in range(min_ix, max_ix + 1):
				center_x = grid_min_x + (ix + 0.5) * res_x
				if _point_in_polygon(center_x, center_y, polygon):
					grid[iy, ix] = 1


def _to_tensor(
	value: Tuple[float, float] | torch.Tensor,
	*,
	device: torch.device | None = None,
) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return value.to(device=device) if device is not None else value
	return torch.tensor(value, dtype=torch.float32, device=device)


def _to_float_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return value.to(dtype=torch.float32)
	return torch.tensor(value, dtype=torch.float32)


def _normalize_obstacles(static_obstacles: Sequence[object]) -> List[List[Tuple[float, float]]]:
	normalized: List[List[Tuple[float, float]]] = []
	for obstacle in static_obstacles:
		if hasattr(obstacle, "vertices"):
			vertices = getattr(obstacle, "vertices")
		else:
			vertices = obstacle

		polygon = [(float(x), float(y)) for x, y in vertices]
		normalized.append(polygon)
	return normalized


def _point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
	"""Ray casting point-in-polygon test."""
	inside = False
	j = len(polygon) - 1
	for i in range(len(polygon)):
		xi, yi = polygon[i]
		xj, yj = polygon[j]
		intersects = (yi > y) != (yj > y)
		if intersects:
			x_intersection = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
			if x < x_intersection:
				inside = not inside
		j = i
	return inside


def main() -> None:
	# Example ORCA-like trajectory: T=80, N=2
	steps = 80
	traj = np.zeros((steps, 2, 2), dtype=np.float32)
	traj[:, 0, 0] = np.linspace(0.5, 7.0, steps)
	traj[:, 0, 1] = 2.0
	traj[:, 1, 0] = 6.5
	traj[:, 1, 1] = np.linspace(7.0, 1.0, steps)

	static_obstacles = [
		[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
		[(3.0, 3.0), (4.5, 3.0), (4.5, 4.5), (3.0, 4.5)],
	]

	occ2d = Occupancy2d(
		resolution=(0.1, 0.1),
		size=(8.0, 8.0),
		trajectory=traj,
		static_obstacles=static_obstacles,
		agent_radius=0.25,
	)
	grids_default = occ2d.generate()
	occ_default = grids_default[-1].cpu().numpy()
	offset = np.array([2.5, -2.0], dtype=np.float32)
	grids_offset = occ2d.generate(center_offset=tuple(offset.tolist()))
	occ_offset = grids_offset[-1].cpu().numpy()

	fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
	ax_default = axes[0, 0]
	ax_offset = axes[0, 1]

	ax_default.imshow(occ_default, origin="lower", cmap="gray_r", vmin=0, vmax=1)
	ax_default.set_title("Default Center")
	ax_default.set_xlabel("X cell")
	ax_default.set_ylabel("Y cell")

	ax_offset.imshow(occ_offset, origin="lower", cmap="gray_r", vmin=0, vmax=1)
	ax_offset.set_title(f"Offset Center (dx={offset[0]:.1f}, dy={offset[1]:.1f})")
	ax_offset.set_xlabel("X cell")
	ax_offset.set_ylabel("Y cell")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
