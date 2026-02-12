"""Manage occupancy data, frontier masks, and coverage masks."""

import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import binary_opening, label
from .utils import world_to_grid, cells_in_radius, bresenham_line


UNKNOWN = -1
FREE = 0
OBSTACLE_THRESHOLD = 50


class MapManager:
    def __init__(self, w: float = 0.15, obstacle_buffer: float = 0.12):
        self.w = w
        self.obstacle_buffer = obstacle_buffer

        self.occupancy_grid: Optional[np.ndarray] = None
        self.sampled_mask: Optional[np.ndarray] = None
        self.covered_mask: Optional[np.ndarray] = None   # physically covered

        self.resolution: float = 0.05
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.grid_width: int = 0
        self.grid_height: int = 0

        self._old_origin_x: float = 0.0
        self._old_origin_y: float = 0.0

        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_yaw: float = 0.0

        self.map_received: bool = False

    def update_map(
        self,
        data,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float
    ):
        """Update the occupancy grid from a new /map message."""
        new_grid = np.array(data, dtype=np.int8).reshape((height, width))

        map_changed = (
            self.occupancy_grid is None
            or width != self.grid_width
            or height != self.grid_height
            or abs(origin_x - self.origin_x) > 1e-6
            or abs(origin_y - self.origin_y) > 1e-6
        )

        if map_changed and self.sampled_mask is not None:
            self.sampled_mask = self._embed_sampled_mask(
                width, height, resolution, origin_x, origin_y
            )
        elif self.sampled_mask is None:
            self.sampled_mask = np.zeros((height, width), dtype=bool)

        if map_changed and self.covered_mask is not None:
            self.covered_mask = self._embed_covered_mask(
                width, height, resolution, origin_x, origin_y
            )
        elif self.covered_mask is None:
            self.covered_mask = np.zeros((height, width), dtype=bool)

        self._old_origin_x = self.origin_x
        self._old_origin_y = self.origin_y

        self.occupancy_grid = new_grid
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.grid_width = width
        self.grid_height = height
        self.map_received = True

    def _embed_sampled_mask(
        self,
        new_width: int,
        new_height: int,
        new_resolution: float,
        new_origin_x: float,
        new_origin_y: float
    ) -> np.ndarray:
        """Re-embed old sampled_mask into new larger array when map resizes."""
        new_mask = np.zeros((new_height, new_width), dtype=bool)
        if self.sampled_mask is None:
            return new_mask

        offset_col = int(round((self.origin_x - new_origin_x) / new_resolution))
        offset_row = int(round((self.origin_y - new_origin_y) / new_resolution))

        old_h, old_w = self.sampled_mask.shape

        src_c0 = max(0, -offset_col)
        src_r0 = max(0, -offset_row)
        src_c1 = min(old_w, new_width - offset_col)
        src_r1 = min(old_h, new_height - offset_row)

        dst_c0 = max(0, offset_col)
        dst_r0 = max(0, offset_row)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)

        if (src_c1 > src_c0 and src_r1 > src_r0
                and dst_c1 <= new_width and dst_r1 <= new_height):
            new_mask[dst_r0:dst_r1, dst_c0:dst_c1] = \
                self.sampled_mask[src_r0:src_r1, src_c0:src_c1]

        return new_mask

    def update_robot_position(self, x: float, y: float, yaw: float):
        """Update robot position from odometry."""
        self.robot_x = x
        self.robot_y = y
        self.robot_yaw = yaw

    def get_robot_position(self) -> Tuple[float, float]:
        return self.robot_x, self.robot_y

    def get_robot_grid_position(self) -> Tuple[int, int]:
        return world_to_grid(
            self.robot_x, self.robot_y,
            self.origin_x, self.origin_y,
            self.resolution
        )

    def is_free(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return False
        return int(self.occupancy_grid[row, col]) == FREE

    def is_unknown(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return True  # out of bounds = unknown
        return int(self.occupancy_grid[row, col]) == UNKNOWN

    def is_obstacle(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return False
        return int(self.occupancy_grid[row, col]) >= OBSTACLE_THRESHOLD

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width

    def get_sampling_front(self) -> np.ndarray:
        """Raw sampling front: FREE ∧ ¬sampled."""
        if self.occupancy_grid is None or self.sampled_mask is None:
            return np.zeros((0, 0), dtype=bool)
        return (self.occupancy_grid == FREE) & (~self.sampled_mask)

    def get_sampling_front_filtered(self) -> np.ndarray:
        """
        Layer 2 noise filter — morphological cleaning of sampling front.

        Removes tiny isolated free-cell patches that SLAM created from
        stray LiDAR rays punching through gaps.
        """
        raw_front = self.get_sampling_front()
        if not np.any(raw_front):
            return raw_front

        cleaned = binary_opening(raw_front, structure=np.ones((3, 3)))

        min_cluster = max(4, int(0.5 * self.w / self.resolution))
        labeled, num_features = label(cleaned)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_cluster:
                cleaned[labeled == i] = False

        return cleaned

    def mark_sampling_front_done(self, sampling_front: np.ndarray):
        """One-way OR: sampled_mask only grows."""
        if self.sampled_mask is not None:
            self.sampled_mask = self.sampled_mask | sampling_front

    def is_frontier_point(
        self, row: int, col: int, boundary_radius_m: Optional[float] = None
    ) -> bool:
        """Does B(s, w) contain ANY unknown or obstacle cell?"""
        if not self.is_free(row, col):
            return False
        if boundary_radius_m is None:
            boundary_radius_m = self.w
        radius_cells = int(np.ceil(boundary_radius_m / self.resolution))

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r, c = row + dr, col + dc
                if not self._in_bounds(r, c):
                    return True  # out of bounds = unknown
                val = int(self.occupancy_grid[r, c])
                if val == UNKNOWN or val >= OBSTACLE_THRESHOLD:
                    return True
        return False

    def is_robust_frontier_point(
        self,
        row: int,
        col: int,
        min_cells: int = 3,
        boundary_radius_m: Optional[float] = None,
    ) -> bool:
        if not self.is_free(row, col):
            return False
        if boundary_radius_m is None:
            boundary_radius_m = self.w
        radius_cells = int(np.ceil(boundary_radius_m / self.resolution))
        count = 0

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r, c = row + dr, col + dc
                if not self._in_bounds(r, c):
                    count += 1  # out of bounds = unknown
                else:
                    val = int(self.occupancy_grid[r, c])
                    if val == UNKNOWN or val >= OBSTACLE_THRESHOLD:
                        count += 1
                if count >= min_cells:
                    return True
        return count >= min_cells

    def is_safe_from_obstacles(self, row: int, col: int) -> bool:
        if self.obstacle_buffer <= 0:
            return True                     # disabled — allow everywhere
        buffer_cells = int(self.obstacle_buffer / self.resolution)
        if buffer_cells < 1:
            return True                     # sub-cell buffer → skip
        for dr in range(-buffer_cells, buffer_cells + 1):
            for dc in range(-buffer_cells, buffer_cells + 1):
                if dr * dr + dc * dc > buffer_cells * buffer_cells:
                    continue                # circular, not square
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    if int(self.occupancy_grid[r, c]) >= OBSTACLE_THRESHOLD:
                        return False
        return True

    def distance_to_nearest_obstacle(
        self, row: int, col: int, search_radius: int = 20
    ) -> float:
        for r in range(1, search_radius + 1):
            for (cr, cc) in cells_in_radius(
                row, col, r, self.grid_height, self.grid_width
            ):
                if self.is_obstacle(cr, cc):
                    return r * self.resolution
        return search_radius * self.resolution

    def is_adjacent_to_unknown(self, row: int, col: int, radius_m: float) -> bool:
        """Is B(point, radius_m) touching any UNKNOWN cell?"""
        radius_cells = int(np.ceil(radius_m / self.resolution))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r, c = row + dr, col + dc
                if not self._in_bounds(r, c):
                    return True  # out of bounds = unknown
                if int(self.occupancy_grid[r, c]) == UNKNOWN:
                    return True
        return False

    def mark_edge_covered(
        self, x1: float, y1: float, x2: float, y2: float
    ):
        if self.covered_mask is None or self.occupancy_grid is None:
            return
        r0, c0 = world_to_grid(
            x1, y1, self.origin_x, self.origin_y, self.resolution
        )
        r1, c1 = world_to_grid(
            x2, y2, self.origin_x, self.origin_y, self.resolution
        )
        line_cells = bresenham_line(r0, c0, r1, c1)
        half_w_cells = max(1, int((self.w / 2.0) / self.resolution))
        h, w_grid = self.covered_mask.shape

        for lr, lc in line_cells:
            rlo = max(0, lr - half_w_cells)
            rhi = min(h, lr + half_w_cells + 1)
            clo = max(0, lc - half_w_cells)
            chi = min(w_grid, lc + half_w_cells + 1)
            for rr in range(rlo, rhi):
                for cc in range(clo, chi):
                    if int(self.occupancy_grid[rr, cc]) >= OBSTACLE_THRESHOLD:
                        continue
                    self.covered_mask[rr, cc] = True

    def is_already_covered(self, row: int, col: int) -> bool:
        """Has cell (row, col) been physically covered by a traversed edge?"""
        if self.covered_mask is None:
            return False
        if not self._in_bounds(row, col):
            return False
        return bool(self.covered_mask[row, col])

    def _embed_covered_mask(
        self,
        new_width: int,
        new_height: int,
        new_resolution: float,
        new_origin_x: float,
        new_origin_y: float,
    ) -> np.ndarray:
        """Re-embed covered_mask into new larger array when map resizes."""
        new_mask = np.zeros((new_height, new_width), dtype=bool)
        if self.covered_mask is None:
            return new_mask

        offset_col = int(round((self.origin_x - new_origin_x) / new_resolution))
        offset_row = int(round((self.origin_y - new_origin_y) / new_resolution))
        old_h, old_w = self.covered_mask.shape

        src_c0 = max(0, -offset_col)
        src_r0 = max(0, -offset_row)
        src_c1 = min(old_w, new_width - offset_col)
        src_r1 = min(old_h, new_height - offset_row)

        dst_c0 = max(0, offset_col)
        dst_r0 = max(0, offset_row)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)

        if (src_c1 > src_c0 and src_r1 > src_r0
                and dst_c1 <= new_width and dst_r1 <= new_height):
            new_mask[dst_r0:dst_r1, dst_c0:dst_c1] = \
                self.covered_mask[src_r0:src_r1, src_c0:src_c1]

        return new_mask
