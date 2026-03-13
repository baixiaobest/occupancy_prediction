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
		center: Tuple[float, float] | torch.Tensor | None = None,
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

		default_center = self.size * 0.5
		if center is not None:
			center_tensor = _to_tensor(center)
		else:
			center_tensor = default_center

		if center_tensor.numel() != 2:
			raise ValueError("center must have exactly two elements")

		self.center = center_tensor.to(dtype=torch.float32)

	def update_inputs(
		self,
		trajectory: np.ndarray | torch.Tensor,
		static_obstacles: Sequence[object] | None = None,
	) -> None:
		self.trajectory = _to_float_tensor(trajectory)
		if static_obstacles is not None:
			self.static_obstacles = _normalize_obstacles(static_obstacles)

	def generate(self) -> List[torch.Tensor]:
		"""Generate a list of occupancy grids (one per timestep).

		Returns:
			List[torch.Tensor]: each grid has shape (H, W), with 1 occupied and 0 free.
		"""
		if self.trajectory is None:
			raise ValueError("trajectory is not set. Call update_inputs() or pass it in __init__.")

		if self.trajectory.ndim != 3 or self.trajectory.shape[-1] != 2:
			raise ValueError("trajectory must have shape (T, N, 2)")

		cells_x = int(torch.floor(self.size[0] / self.resolution[0]).item())
		cells_y = int(torch.floor(self.size[1] / self.resolution[1]).item())

		if cells_x <= 0 or cells_y <= 0:
			raise ValueError("size/resolution yields non-positive grid dimensions")

		trajectory = self.trajectory.to(dtype=torch.float32)
		device = trajectory.device

		grids: List[torch.Tensor] = []
		for t in range(trajectory.shape[0]):
			grid = torch.zeros((cells_y, cells_x), dtype=torch.uint8, device=device)

			# Static obstacles are present in every frame.
			for polygon in self.static_obstacles:
				self._rasterize_polygon(grid, polygon)

			# Dynamic occupancy from agent positions at this timestep.
			for n in range(trajectory.shape[1]):
				self._rasterize_agent(grid, trajectory[t, n])

			grids.append(grid)

		return grids

	def _rasterize_agent(self, grid: torch.Tensor, position: torch.Tensor) -> None:
		half_size = torch.tensor(
			[self.agent_radius, self.agent_radius], device=grid.device, dtype=torch.float32
		)
		self._rasterize_box(grid, position.to(grid.device), half_size)

	def _rasterize_box(
		self, grid: torch.Tensor, center_xy: torch.Tensor, half_size_xy: torch.Tensor
	) -> None:
		cells_y, cells_x = grid.shape
		resolution = self.resolution.to(device=grid.device, dtype=torch.float32)
		grid_min = (self.center - self.size * 0.5).to(device=grid.device, dtype=torch.float32)

		min_xy = center_xy - half_size_xy
		max_xy = center_xy + half_size_xy

		min_idx = torch.floor((min_xy - grid_min) / resolution).to(dtype=torch.long)
		max_idx = torch.floor((max_xy - grid_min) / resolution).to(dtype=torch.long)

		min_ix = int(torch.clamp(min_idx[0], 0, cells_x - 1).item())
		max_ix = int(torch.clamp(max_idx[0], 0, cells_x - 1).item())
		min_iy = int(torch.clamp(min_idx[1], 0, cells_y - 1).item())
		max_iy = int(torch.clamp(max_idx[1], 0, cells_y - 1).item())

		if min_ix > max_ix or min_iy > max_iy:
			return

		grid[min_iy : max_iy + 1, min_ix : max_ix + 1] = 1

	def _rasterize_polygon(self, grid: torch.Tensor, polygon: List[Tuple[float, float]]) -> None:
		if len(polygon) < 3:
			return

		cells_y, cells_x = grid.shape
		polygon_np = np.asarray(polygon, dtype=np.float32)

		min_x = float(np.min(polygon_np[:, 0]))
		max_x = float(np.max(polygon_np[:, 0]))
		min_y = float(np.min(polygon_np[:, 1]))
		max_y = float(np.max(polygon_np[:, 1]))

		res_x = float(self.resolution[0].item())
		res_y = float(self.resolution[1].item())
		grid_min_x = float((self.center[0] - self.size[0] * 0.5).item())
		grid_min_y = float((self.center[1] - self.size[1] * 0.5).item())

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
	grids = occ2d.generate()
	occ = grids[-1].cpu().numpy()

	plt.figure(figsize=(6, 6))
	plt.imshow(occ, origin="lower", cmap="gray_r")
	plt.title("Occupancy Grid")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
