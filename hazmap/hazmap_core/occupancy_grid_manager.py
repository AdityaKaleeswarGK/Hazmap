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

        # ── Next-Best-View observation model ──────────────────────────────
        # Per-cell best sensing quality (0..1) with which each cell has been
        # *seen* (line-of-sight, within sensor range). Distinct from
        # _covered_map (cells driven over). Rebuilt from the recorded
        # viewpoint history on map resize/origin shift, like _covered_map.
        self._observation_quality: Optional[np.ndarray] = None
        self._observation_views: List[Tuple[float, float, float, float, int]] = []

        # ── Surface-inspection model ──────────────────────────────────────
        # Per-OBSTACLE-cell best inspection quality (0..1) with which each
        # obstacle surface cell has been *viewed* from within the camera's
        # usable range band [min_view, max_view] and with line of sight.
        # Distinct from _observation_quality (which records free cells seen).
        # Drives the "orbit obstacles and view all faces" behavior: a
        # candidate that would newly view un-inspected surface scores higher.
        # Rebuilt from recorded surface-view history on map resize, like the
        # other two grids.
        self._surface_seen: Optional[np.ndarray] = None
        self._surface_views: List[Tuple[float, float, float, float, float, int]] = []

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
            self._rebuild_observation_quality()
            self._rebuild_surface_seen()
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

    # ------------------------------------------------------------------
    # Next-Best-View observation model
    # ------------------------------------------------------------------
    def _rebuild_observation_quality(self) -> None:
        """Allocate a fresh quality grid and replay the viewpoint history."""
        if self._data is None:
            self._observation_quality = None
            return
        self._observation_quality = np.zeros(
            (self._height, self._width), dtype=np.float32
        )
        for vx, vy, vr, vfov, vrays in self._observation_views:
            self._cast_observation(vx, vy, vr, vfov, vrays, commit=True)

    def observe_from(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        fov_deg: float = 360.0,
        n_rays: int = 72,
        yaw: float = 0.0,
        record: bool = True,
    ) -> None:
        """Raycast from a viewpoint and raise the observation quality of every
        unoccluded free cell it sees. Quality decays linearly with distance."""
        if record:
            self._observation_views.append(
                (wx, wy, sensor_range, fov_deg, n_rays)
            )
        if self._data is None:
            return  # map not ready yet; replayed on first update()
        if (
            self._observation_quality is None
            or self._observation_quality.shape != (self._height, self._width)
        ):
            # (Re)allocate and replay history (current view already recorded).
            self._rebuild_observation_quality()
            return
        self._cast_observation(wx, wy, sensor_range, fov_deg, n_rays,
                               commit=True, yaw=yaw)

    def predict_observation_gain(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        fov_deg: float = 360.0,
        n_rays: int = 72,
        yaw: float = 0.0,
    ) -> float:
        """Quality-weighted info gain a viewpoint would add WITHOUT committing:
        sum over visible free cells of max(0, q_new - q_current)."""
        if self._data is None:
            return 0.0
        return self._cast_observation(wx, wy, sensor_range, fov_deg, n_rays,
                                      commit=False, yaw=yaw)

    def _cast_observation(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        fov_deg: float,
        n_rays: int,
        commit: bool,
        yaw: float = 0.0,
    ) -> float:
        """Shared raycast core. If commit, writes max-quality into the grid and
        returns 0. If not commit, returns the predicted added quality (gain)."""
        if self._data is None:
            return 0.0
        gx0, gy0 = self.world_to_grid(wx, wy)
        if not self._in_bounds(gx0, gy0):
            return 0.0
        range_cells = max(1, int(sensor_range / self._resolution))
        n_rays = max(1, int(n_rays))
        full_circle = fov_deg >= 359.0
        fov = math.radians(fov_deg)

        # Dedup cells across rays within this single viewpoint (keep best q).
        seen: dict = {} if not commit else None

        for i in range(n_rays):
            if full_circle:
                ang = yaw + 2.0 * math.pi * (i / n_rays)
            elif n_rays > 1:
                ang = yaw - fov / 2.0 + fov * (i / (n_rays - 1))
            else:
                ang = yaw
            ex = gx0 + int(round(range_cells * math.cos(ang)))
            ey = gy0 + int(round(range_cells * math.sin(ang)))
            for gx, gy in self._bresenham(gx0, gy0, ex, ey):
                if gx == gx0 and gy == gy0:
                    continue
                if not self._in_bounds(gx, gy):
                    break
                d = math.hypot(gx - gx0, gy - gy0) * self._resolution
                if d > sensor_range:
                    break
                v = int(self._data[gy, gx])
                if v == self.UNKNOWN:
                    break  # cannot see past unknown space
                if v >= self._free_threshold:
                    break  # occluded by obstacle
                q = 1.0 - d / sensor_range
                if q <= 0.0:
                    continue
                if commit:
                    if q > self._observation_quality[gy, gx]:
                        self._observation_quality[gy, gx] = q
                else:
                    prev = seen.get((gy, gx))
                    if prev is None or q > prev:
                        seen[(gy, gx)] = q

        if commit:
            return 0.0
        gain = 0.0
        for (gy, gx), q in seen.items():
            cur = (
                float(self._observation_quality[gy, gx])
                if self._observation_quality is not None
                else 0.0
            )
            if q > cur:
                gain += q - cur
        return gain

    # ------------------------------------------------------------------
    # Surface inspection (view obstacle faces from the camera range band)
    # ------------------------------------------------------------------
    def _rebuild_surface_seen(self) -> None:
        """Allocate a fresh surface-quality grid and replay the view history."""
        if self._data is None:
            self._surface_seen = None
            return
        self._surface_seen = np.zeros(
            (self._height, self._width), dtype=np.float32
        )
        for vx, vy, vr, vmin, vmax, vrays in self._surface_views:
            self._cast_surface(vx, vy, vr, vmin, vmax, vrays, commit=True)

    def mark_surface_seen(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        min_view: float,
        max_view: float,
        n_rays: int = 72,
    ) -> None:
        """Raycast from a viewpoint and record every obstacle surface cell
        hit within the usable camera band [min_view, max_view] (with line of
        sight). Quality peaks mid-band and falls off toward the edges, so the
        selector prefers viewpoints that frame a surface at a comfortable
        distance rather than grazing it."""
        self._surface_views.append(
            (wx, wy, sensor_range, min_view, max_view, n_rays)
        )
        if self._data is None:
            return  # map not ready; replayed on first update()
        if (
            self._surface_seen is None
            or self._surface_seen.shape != (self._height, self._width)
        ):
            self._rebuild_surface_seen()
            return
        self._cast_surface(
            wx, wy, sensor_range, min_view, max_view, n_rays, commit=True
        )

    def predict_surface_gain(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        min_view: float,
        max_view: float,
        n_rays: int = 72,
    ) -> float:
        """New surface-inspection quality a viewpoint would add WITHOUT
        committing: sum over viewable obstacle cells of max(0, q_new - q_cur).
        Zero once every surface this viewpoint can see is already inspected —
        which is what makes the rover move on / orbit to an un-inspected face."""
        if self._data is None:
            return 0.0
        return self._cast_surface(
            wx, wy, sensor_range, min_view, max_view, n_rays, commit=False
        )

    def _cast_surface(
        self,
        wx: float,
        wy: float,
        sensor_range: float,
        min_view: float,
        max_view: float,
        n_rays: int,
        commit: bool,
    ) -> float:
        """Shared surface raycast. Walks rays until they hit an obstacle (or
        unknown / range limit). An obstacle cell hit at distance d within
        [min_view, max_view] is the inspectable surface; its quality is a
        triangular falloff that peaks at the band midpoint. 360° because the
        rover can rotate the camera in place at a viewpoint."""
        if self._data is None:
            return 0.0
        gx0, gy0 = self.world_to_grid(wx, wy)
        if not self._in_bounds(gx0, gy0):
            return 0.0
        cap = min(sensor_range, max_view)
        range_cells = max(1, int(cap / self._resolution))
        n_rays = max(1, int(n_rays))
        mid = 0.5 * (min_view + max_view)
        half_span = max(1e-3, 0.5 * (max_view - min_view))

        seen: dict = {} if not commit else None
        for i in range(n_rays):
            ang = 2.0 * math.pi * (i / n_rays)
            ex = gx0 + int(round(range_cells * math.cos(ang)))
            ey = gy0 + int(round(range_cells * math.sin(ang)))
            for gx, gy in self._bresenham(gx0, gy0, ex, ey):
                if gx == gx0 and gy == gy0:
                    continue
                if not self._in_bounds(gx, gy):
                    break
                d = math.hypot(gx - gx0, gy - gy0) * self._resolution
                if d > cap:
                    break
                v = int(self._data[gy, gx])
                if v == self.UNKNOWN:
                    break  # cannot see past unknown space
                if v >= self._free_threshold:
                    # Obstacle surface cell. Credit it only if it sits within
                    # the usable band; too close (out of focus / framing) or
                    # beyond range does not count as a good inspection view.
                    if d >= min_view:
                        q = 1.0 - abs(d - mid) / half_span
                        if q > 0.0:
                            if commit:
                                if q > self._surface_seen[gy, gx]:
                                    self._surface_seen[gy, gx] = q
                            else:
                                prev = seen.get((gy, gx))
                                if prev is None or q > prev:
                                    seen[(gy, gx)] = q
                    break  # ray stops at the obstacle either way
        if commit:
            return 0.0
        gain = 0.0
        for (gy, gx), q in seen.items():
            cur = (
                float(self._surface_seen[gy, gx])
                if self._surface_seen is not None
                else 0.0
            )
            if q > cur:
                gain += q - cur
        return gain

    def surface_quality_grid_int8(self) -> Optional[np.ndarray]:
        """Surface-inspection quality scaled to 0..100 int8 for RViz."""
        if self._surface_seen is None:
            return None
        scaled = np.clip(self._surface_seen * 100.0, 0, 100)
        return scaled.astype(np.int8)

    def get_surface_statistics(
        self, q_min: float, view_radius: float
    ) -> Tuple[float, float, float]:
        """Surface-inspection coverage: (percent, inspected_cells,
        total_inspectable_cells). "Inspectable" = obstacle cells adjacent to
        known free space (i.e. surfaces a camera could ever reach), within
        view_radius logic handled by the caller. A surface counts inspected
        once its quality reaches q_min."""
        if self._data is None or self._surface_seen is None:
            return 0.0, 0.0, 0.0
        obstacles = self._data >= self._free_threshold
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        # Obstacle cells adjacent to free space = reachable surfaces.
        adj = np.zeros_like(obstacles)
        adj[1:, :] |= frees[:-1, :]
        adj[:-1, :] |= frees[1:, :]
        adj[:, 1:] |= frees[:, :-1]
        adj[:, :-1] |= frees[:, 1:]
        inspectable = obstacles & adj
        total = float(np.sum(inspectable))
        if total == 0:
            return 0.0, 0.0, 0.0
        inspected = float(np.sum(inspectable & (self._surface_seen >= q_min)))
        return 100.0 * inspected / total, inspected, total

    def get_observation_statistics(
        self, q_min: float
    ) -> Tuple[float, float, float]:
        """Observation coverage: (percent, observed_m2, total_free_m2), where a
        free cell counts as observed once its quality reaches q_min."""
        if self._data is None or self._observation_quality is None:
            return 0.0, 0.0, 0.0
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        total_free_cells = np.sum(frees)
        if total_free_cells == 0:
            return 0.0, 0.0, 0.0
        observed = frees & (self._observation_quality >= q_min)
        observed_cells = np.sum(observed)
        percent = (observed_cells / total_free_cells) * 100.0
        area_m2 = observed_cells * (self._resolution**2)
        total_m2 = total_free_cells * (self._resolution**2)
        return percent, area_m2, total_m2

    def observation_quality_grid_int8(self) -> Optional[np.ndarray]:
        """Quality grid scaled to 0..100 int8 for OccupancyGrid visualization."""
        if self._observation_quality is None:
            return None
        scaled = np.clip(self._observation_quality * 100.0, 0, 100)
        return scaled.astype(np.int8)

    def predict_coverage_gain(
        self,
        wx: float,
        wy: float,
        radius: float,
        lambda_unknown: float = 0.0,
    ) -> float:
        """Information-aware gain: uncovered FREE cells in the disk plus
        lambda_unknown times the unknown-boundary cells (free↔unknown
        adjacencies) inside the disk. With lambda_unknown=0 this reduces to
        a pure swath count (legacy behavior). With lambda_unknown>0 the
        selector also rewards candidates sitting on an unknown frontier —
        i.e. places where moving the rover *reveals* new map."""
        if self._data is None:
            return 0.0
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = max(1, int(radius / self._resolution))
        x_min = max(0, gx0 - r_cells)
        x_max = min(self._width, gx0 + r_cells + 1)
        y_min = max(0, gy0 - r_cells)
        y_max = min(self._height, gy0 + r_cells + 1)
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        sub = self._data[y_min:y_max, x_min:x_max]
        free_mask = (sub >= 0) & (sub < self._free_threshold)
        if self._covered_map is not None:
            free_mask &= ~self._covered_map[y_min:y_max, x_min:x_max]
        dy = np.arange(y_min, y_max) - gy0
        dx = np.arange(x_min, x_max) - gx0
        disk = (dy[:, None] ** 2 + dx[None, :] ** 2) <= r_cells * r_cells
        swath = float(np.sum(free_mask & disk))
        if lambda_unknown <= 0.0:
            return swath
        # Unknown-boundary cells: free cells adjacent to UNKNOWN, restricted
        # to this disk. Reuse the global frontier mask so logic stays in one
        # place (and the OGM keeps a single definition of "frontier").
        fmask = self._frontier_mask()
        if fmask is None:
            return swath
        unknown_boundary = float(
            np.sum(fmask[y_min:y_max, x_min:x_max] & disk)
        )
        return swath + lambda_unknown * unknown_boundary

    def find_frontier_cells_in_disk(
        self, wx: float, wy: float, radius: float
    ) -> int:
        """Count uncovered free↔unknown boundary cells inside a disk —
        used by RCG.reopen_stale_closed to detect when the post-close map
        update has surfaced more to explore near a CLOSED node."""
        if self._data is None:
            return 0
        fmask = self._frontier_mask()
        if fmask is None:
            return 0
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = max(1, int(radius / self._resolution))
        x_min = max(0, gx0 - r_cells)
        x_max = min(self._width, gx0 + r_cells + 1)
        y_min = max(0, gy0 - r_cells)
        y_max = min(self._height, gy0 + r_cells + 1)
        if x_min >= x_max or y_min >= y_max:
            return 0
        dy = np.arange(y_min, y_max) - gy0
        dx = np.arange(x_min, x_max) - gx0
        disk = (dy[:, None] ** 2 + dx[None, :] ** 2) <= r_cells * r_cells
        return int(np.sum(fmask[y_min:y_max, x_min:x_max] & disk))

    def local_complexity(
        self,
        wx: float,
        wy: float,
        radius: float,
        obstacle_weight: float = 0.5,
        boundary_weight: float = 0.5,
    ) -> float:
        """Local environmental complexity C ∈ [0, 1] used to modulate
        sampling density. High in cluttered or frontier-rich regions
        (obstacles nearby, lots of unknown boundary to inspect), low in
        big open free zones. Result drives density-modulated keep in the
        sampler — dense nodes where it matters, sparse where it doesn't."""
        if self._data is None:
            return 0.0
        gx0, gy0 = self.world_to_grid(wx, wy)
        r_cells = max(1, int(radius / self._resolution))
        x_min = max(0, gx0 - r_cells)
        x_max = min(self._width, gx0 + r_cells + 1)
        y_min = max(0, gy0 - r_cells)
        y_max = min(self._height, gy0 + r_cells + 1)
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        sub = self._data[y_min:y_max, x_min:x_max]
        dy = np.arange(y_min, y_max) - gy0
        dx = np.arange(x_min, x_max) - gx0
        disk = (dy[:, None] ** 2 + dx[None, :] ** 2) <= r_cells * r_cells
        disk_area = float(np.sum(disk))
        if disk_area <= 0.0:
            return 0.0
        obstacle_mask = sub >= self._free_threshold
        obstacle_density = float(np.sum(obstacle_mask & disk)) / disk_area
        fmask = self._frontier_mask()
        if fmask is not None:
            boundary_density = float(
                np.sum(fmask[y_min:y_max, x_min:x_max] & disk)
            ) / disk_area
        else:
            boundary_density = 0.0
        # Obstacle density saturates fast (a wall in view ≈ "cluttered"); use
        # a 3x scaling so 33% obstacles in the disk → C contribution = 1.0
        # before clamping. Boundary density is rarer; keep linear-ish but
        # scale ×6 since frontier ribbons are thin.
        c_obs = min(1.0, 3.0 * obstacle_density)
        c_bnd = min(1.0, 6.0 * boundary_density)
        total = obstacle_weight + boundary_weight
        if total <= 0.0:
            return 0.0
        return min(1.0, (obstacle_weight * c_obs + boundary_weight * c_bnd) / total)

    # ------------------------------------------------------------------
    # Frontier clustering (reach non-lap-aligned openings)
    # ------------------------------------------------------------------
    def _frontier_mask(self) -> Optional[np.ndarray]:
        """Boolean grid of uncovered FREE cells adjacent to UNKNOWN space."""
        if self._data is None:
            return None
        unknowns = self._data == self.UNKNOWN
        frees = (self._data >= 0) & (self._data < self._free_threshold)
        if self._covered_map is not None:
            frees = frees & (~self._covered_map)
        dil = np.zeros_like(unknowns)
        dil[1:, :] |= unknowns[:-1, :]
        dil[:-1, :] |= unknowns[1:, :]
        dil[:, 1:] |= unknowns[:, :-1]
        dil[:, :-1] |= unknowns[:, 1:]
        return frees & dil

    def find_frontier_clusters(
        self,
        min_cluster_cells: int = 3,
        exclude: Optional[List[Tuple[float, float]]] = None,
        exclude_radius: float = 0.0,
    ) -> List[Tuple[float, float, int]]:
        """Group frontier cells into 8-connected clusters and return one
        navigation target per cluster (the cluster cell nearest its centroid).

        Clusters are found on the raw grid, so openings that the lap-grid
        sampler misses (narrow, oblique, between-lap) still yield a target.
        Returns [(wx, wy, cell_count), ...]; excludes clusters whose target is
        within exclude_radius of any blacklisted point."""
        frontier = self._frontier_mask()
        if frontier is None or not np.any(frontier):
            return []
        ys, xs = np.where(frontier)
        cell_set = set(zip(ys.tolist(), xs.tolist()))
        visited: set = set()
        neighbors8 = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        clusters: List[Tuple[float, float, int]] = []
        for cell in cell_set:
            if cell in visited:
                continue
            stack = [cell]
            visited.add(cell)
            comp: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in neighbors8:
                    nb = (cy + dy, cx + dx)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            if len(comp) < min_cluster_cells:
                continue
            arr = np.array(comp)
            cy_mean = arr[:, 0].mean()
            cx_mean = arr[:, 1].mean()
            d2 = (arr[:, 0] - cy_mean) ** 2 + (arr[:, 1] - cx_mean) ** 2
            best = comp[int(np.argmin(d2))]
            row, col = best
            wx, wy = self.grid_to_world(int(col), int(row))
            if exclude and any(
                math.hypot(wx - ex, wy - ey) < exclude_radius
                for ex, ey in exclude
            ):
                continue
            clusters.append((wx, wy, len(comp)))
        return clusters

    def find_nearest_frontier_cluster(
        self,
        wx: float,
        wy: float,
        min_cluster_cells: int = 3,
        min_distance: float = 0.0,
        max_range: float = 1e9,
        exclude: Optional[List[Tuple[float, float]]] = None,
        exclude_radius: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """Nearest non-blacklisted frontier-cluster target to (wx, wy)."""
        clusters = self.find_frontier_clusters(
            min_cluster_cells=min_cluster_cells,
            exclude=exclude,
            exclude_radius=exclude_radius,
        )
        best = None
        best_d = float('inf')
        for cx, cy, _size in clusters:
            d = math.hypot(cx - wx, cy - wy)
            if d < min_distance or d > max_range:
                continue
            if d < best_d:
                best_d = d
                best = (cx, cy)
        return best
