"""
Occupancy Grid Manager for C* Algorithm.

Wraps the SLAM-generated OccupancyGrid to provide convenient spatial queries:
free / obstacle / unknown classification, coordinate transforms, collision
checking, and nearest-obstacle distance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np


class OccupancyGridManager:
    """Thread-safe wrapper around nav_msgs/OccupancyGrid."""

    # OccupancyGrid cell values
    UNKNOWN = -1

    def __init__(self, free_threshold: int = 50):
        self._data: Optional[np.ndarray] = None
        self._resolution: float = 0.05
        self._width: int = 0
        self._height: int = 0
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._free_threshold = free_threshold

        self._covered_map: Optional[np.ndarray] = None
        self._all_painted_centers: List[Tuple[float, float, float]] = []
        self._pending_paints: List[Tuple[float, float, float]] = []

    # ------------------------------------------------------------------
    # Update from ROS message
    # ------------------------------------------------------------------
    def update(self, msg) -> None:
        """Update internal grid from a nav_msgs/OccupancyGrid message.

        Detects map origin / resolution shifts (e.g. caused by slam_toolbox
        loop closures or pose-graph optimisation) and rebuilds the
        coverage bitmap from the world-coord painted-center history so that
        coverage stats stay aligned with the new map frame.
        """
        new_resolution = msg.info.resolution
        new_width = msg.info.width
        new_height = msg.info.height
        new_origin_x = msg.info.origin.position.x
        new_origin_y = msg.info.origin.position.y

        size_changed = (
            self._covered_map is None
            or self._covered_map.shape != (new_height, new_width)
        )
        origin_shifted = (
            self._data is not None
            and (
                abs(new_origin_x - self._origin_x) > 1e-4
                or abs(new_origin_y - self._origin_y) > 1e-4
                or abs(new_resolution - self._resolution) > 1e-6
            )
        )

        self._resolution = new_resolution
        self._width = new_width
        self._height = new_height
        self._origin_x = new_origin_x
        self._origin_y = new_origin_y
        self._data = np.array(msg.data, dtype=np.int8).reshape(
            (new_height, new_width)
        )

        if size_changed or origin_shifted:
            # Re-paint covered cells in the new grid frame from stored
            # world-coord centers, so coverage stats survive map jumps.
            self._rebuild_covered_map()
        else:
            for wx, wy, r in self._pending_paints:
                self._paint_circle_on_map(wx, wy, r)
        self._pending_paints.clear()

    def mark_covered(self, wx: float, wy: float, radius: float) -> None:
        """Record a covered circular volume (incremental when possible)."""
        self._all_painted_centers.append((wx, wy, radius))
        if self._covered_map is None:
            self._pending_paints.append((wx, wy, radius))
        else:
            self._paint_circle_on_map(wx, wy, radius)

    def _rebuild_covered_map(self) -> None:
        self._covered_map = np.zeros((self._height, self._width), dtype=bool)
        for wx, wy, r in self._all_painted_centers:
            self._paint_circle_on_map(wx, wy, r)

    def _paint_circle_on_map(self, wx: float, wy: float, radius: float) -> None:
        gx, gy = self.world_to_grid(wx, wy)
        r_cells = int(radius / self._resolution)

        y_min = max(0, gy - r_cells)
        y_max = min(self._height, gy + r_cells + 1)
        x_min = max(0, gx - r_cells)
        x_max = min(self._width, gx + r_cells + 1)

        if y_min >= y_max or x_min >= x_max:
            return

        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - gx) ** 2 + (y - gy) ** 2 <= r_cells**2
        self._covered_map[y_min:y_max, x_min:x_max][mask] = True

    @property
    def ready(self) -> bool:
        return self._data is not None

    @property
    def resolution(self) -> float:
        return self._resolution

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """World (metres) → grid (col, row)."""
        gx = int((wx - self._origin_x) / self._resolution)
        gy = int((wy - self._origin_y) / self._resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Grid (col, row) → world (metres), centre of cell."""
        wx = self._origin_x + (gx + 0.5) * self._resolution
        wy = self._origin_y + (gy + 0.5) * self._resolution
        return wx, wy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._width and 0 <= gy < self._height

    # ------------------------------------------------------------------
    # Cell classification
    # ------------------------------------------------------------------
    def _cell_value(self, gx: int, gy: int) -> int:
        if not self._in_bounds(gx, gy):
            return self.UNKNOWN
        return int(self._data[gy, gx])

    def is_free(self, wx: float, wy: float) -> bool:
        gx, gy = self.world_to_grid(wx, wy)
        v = self._cell_value(gx, gy)
        return 0 <= v < self._free_threshold

    def is_obstacle(self, wx: float, wy: float) -> bool:
        gx, gy = self.world_to_grid(wx, wy)
        v = self._cell_value(gx, gy)
        return v >= self._free_threshold

    def is_unknown(self, wx: float, wy: float) -> bool:
        gx, gy = self.world_to_grid(wx, wy)
        v = self._cell_value(gx, gy)
        return v == self.UNKNOWN

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------
    def is_collision_free(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Bresenham-style line check between two world points."""
        if not self.ready:
            return False
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)
        for gx, gy in self._bresenham(gx1, gy1, gx2, gy2):
            v = self._cell_value(gx, gy)
            if v >= self._free_threshold or v == self.UNKNOWN:
                return False
        return True

    def is_adjacent_to_unknown(self, wx: float, wy: float, radius: float) -> bool:
        """Check if any cell within *radius* of (wx,wy) is unknown."""
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = max(1, int(radius / self._resolution))
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells:
                    continue
                v = self._cell_value(gx0 + dx, gy0 + dy)
                if v == self.UNKNOWN:
                    return True
        return False

    def is_adjacent_to_obstacle(self, wx: float, wy: float, radius: float) -> bool:
        """Check if any cell within *radius* of (wx,wy) is an obstacle."""
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = max(1, int(radius / self._resolution))
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells:
                    continue
                v = self._cell_value(gx0 + dx, gy0 + dy)
                if v >= self._free_threshold:
                    return True
        return False

    def nearest_obstacle_distance(
        self, wx: float, wy: float, max_range: float = 3.0
    ) -> float:
        """Distance to nearest obstacle cell, capped at *max_range* (vectorized)."""
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = int(max_range / self._resolution)

        x_min = max(0, gx0 - r_cells)
        x_max = min(self._width, gx0 + r_cells + 1)
        y_min = max(0, gy0 - r_cells)
        y_max = min(self._height, gy0 + r_cells + 1)

        if x_min >= x_max or y_min >= y_max:
            return max_range

        grid_slice = self._data[y_min:y_max, x_min:x_max]
        obstacles = grid_slice >= self._free_threshold

        if not np.any(obstacles):
            return max_range

        dx_arr = np.arange(x_min, x_max) - gx0
        dy_arr = np.arange(y_min, y_max) - gy0
        dist_sq = dy_arr[:, None] ** 2 + dx_arr[None, :] ** 2

        masked_dist_sq = np.where(obstacles, dist_sq, np.inf)
        min_dist_sq = np.min(masked_dist_sq)

        if np.isinf(min_dist_sq):
            return max_range

        return float(np.sqrt(min_dist_sq)) * self._resolution

    def get_free_cells_in_radius(
        self, wx: float, wy: float, radius: float
    ) -> List[Tuple[float, float]]:
        """Return world-coordinate centres of all free cells within radius."""
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = int(radius / self._resolution)
        result: List[Tuple[float, float]] = []
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells:
                    continue
                gx, gy = gx0 + dx, gy0 + dy
                v = self._cell_value(gx, gy)
                if 0 <= v < self._free_threshold:
                    result.append(self.grid_to_world(gx, gy))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        """Yield integer coordinates on the line from (x0,y0) to (x1,y1)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ------------------------------------------------------------------
    # Frontier detection for coverage completion check
    # ------------------------------------------------------------------
    def has_frontier_cells(self, check_radius: float = 0.5) -> bool:
        """
        Check if there are any free cells adjacent to unknown cells
        in the entire map using fast vectorized NumPy arrays.
        """
        if self._data is None:
            return False

        # Find unknown and free cells
        unknowns = self._data == self.UNKNOWN
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        if self._covered_map is not None:
            frees &= ~self._covered_map

        # Dilate unknown cells by 1 pixel in 4-conn
        unknown_dilated = np.zeros_like(unknowns)
        unknown_dilated[1:, :] |= unknowns[:-1, :]
        unknown_dilated[:-1, :] |= unknowns[1:, :]
        unknown_dilated[:, 1:] |= unknowns[:, :-1]
        unknown_dilated[:, :-1] |= unknowns[:, 1:]

        # A frontier is any free cell that overlaps an expanded unknown region
        frontier_mask = frees & unknown_dilated
        return bool(np.any(frontier_mask))

    def find_nearest_frontier(
        self,
        wx: float,
        wy: float,
        check_radius: float = 0.5,
        max_range: float = 10.0,
        min_distance: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Find the nearest free cell that is adjacent to unknown space
        using fast vectorized Euclidean distance.
        """
        if self._data is None:
            return None

        unknowns = self._data == self.UNKNOWN
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        if self._covered_map is not None:
            frees &= ~self._covered_map

        unknown_dilated = np.zeros_like(unknowns)
        unknown_dilated[1:, :] |= unknowns[:-1, :]
        unknown_dilated[:-1, :] |= unknowns[1:, :]
        unknown_dilated[:, 1:] |= unknowns[:, :-1]
        unknown_dilated[:, :-1] |= unknowns[:, 1:]

        frontier_mask = frees & unknown_dilated
        gy_indices, gx_indices = np.where(frontier_mask)

        if len(gy_indices) == 0:
            return None

        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = int(max_range / self._resolution)
        min_cells = int(min_distance / self._resolution)

        # Bounding box filter for maximum range
        valid_idx = (
            (gx_indices >= gx0 - r_cells)
            & (gx_indices <= gx0 + r_cells)
            & (gy_indices >= gy0 - r_cells)
            & (gy_indices <= gy0 + r_cells)
        )

        gy_indices = gy_indices[valid_idx]
        gx_indices = gx_indices[valid_idx]

        if len(gy_indices) == 0:
            return None

        # Filter by radial distance threshold
        dist_sq = (gx_indices - gx0) ** 2 + (gy_indices - gy0) ** 2
        valid_dist = (dist_sq >= min_cells**2) & (dist_sq <= r_cells**2)
        dist_sq = dist_sq[valid_dist]
        gy_indices = gy_indices[valid_dist]
        gx_indices = gx_indices[valid_dist]

        if len(dist_sq) == 0:
            return None

        best_idx = np.argmin(dist_sq)
        best_gx = gx_indices[best_idx]
        best_gy = gy_indices[best_idx]

        return self.grid_to_world(int(best_gx), int(best_gy))

    def find_nearest_uncovered_free(
        self,
        wx: float,
        wy: float,
        max_range: float = 10.0,
        min_distance: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Find nearest FREE cell that has not yet been marked covered.
        Useful in known-map mode where frontier (unknown-adjacent) logic
        is not the right completion criterion.
        """
        if self._data is None or self._covered_map is None:
            return None

        free_mask = (self._data >= 0) & (self._data < self._free_threshold)
        uncovered_free = free_mask & (~self._covered_map)
        gy_indices, gx_indices = np.where(uncovered_free)

        if len(gy_indices) == 0:
            return None

        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = int(max_range / self._resolution)
        min_cells = int(min_distance / self._resolution)

        valid_idx = (
            (gx_indices >= gx0 - r_cells)
            & (gx_indices <= gx0 + r_cells)
            & (gy_indices >= gy0 - r_cells)
            & (gy_indices <= gy0 + r_cells)
        )
        gy_indices = gy_indices[valid_idx]
        gx_indices = gx_indices[valid_idx]
        if len(gy_indices) == 0:
            return None

        dist_sq = (gx_indices - gx0) ** 2 + (gy_indices - gy0) ** 2
        valid_dist = (dist_sq >= min_cells**2) & (dist_sq <= r_cells**2)
        dist_sq = dist_sq[valid_dist]
        gy_indices = gy_indices[valid_dist]
        gx_indices = gx_indices[valid_dist]
        if len(dist_sq) == 0:
            return None

        best_idx = np.argmin(dist_sq)
        return self.grid_to_world(int(gx_indices[best_idx]), int(gy_indices[best_idx]))

    def get_coverage_statistics(self) -> Tuple[float, float, float]:
        """
        Calculate coverage efficiency.
        Returns (percent_covered, area_covered_m2, total_free_m2).
        """
        if self._data is None or self._covered_map is None:
            return 0.0, 0.0, 0.0

        # Total free cells (habitable space)
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        total_free_cells = np.sum(frees)

        # Area that is both FREE and marked COVERED
        covered_frees = frees & self._covered_map
        covered_cells = np.sum(covered_frees)

        if total_free_cells == 0:
            return 0.0, 0.0, 0.0

        percent = (covered_cells / total_free_cells) * 100.0
        area_m2 = covered_cells * (self._resolution**2)
        total_m2 = total_free_cells * (self._resolution**2)

        return percent, area_m2, total_m2
