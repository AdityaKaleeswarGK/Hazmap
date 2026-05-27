"""Boustrophedon (lawnmower) coverage planner for known occupancy maps."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class BoustrophedonPlanner:
    """Generate sweep endpoints across free-space strips."""

    def __init__(
        self,
        ogm,
        lap_spacing_m: float = 0.7,
        min_segment_m: float = 0.4,
        boundary_margin_m: float = 0.7,
    ):
        self.ogm = ogm
        self.lap_spacing_m = max(0.1, float(lap_spacing_m))
        self.min_segment_m = max(0.1, float(min_segment_m))
        self.boundary_margin_m = max(0.0, float(boundary_margin_m))

    def plan(self, sweep_axis: str = 'x') -> List[Tuple[float, float]]:
        if not self.ogm.ready or self.ogm._data is None:
            return []

        res = float(self.ogm.resolution)
        step = max(1, int(round(self.lap_spacing_m / res)))
        min_len_cells = max(1, int(round(self.min_segment_m / res)))
        margin_cells = int(round(self.boundary_margin_m / res))

        data = self.ogm._data
        free = np.logical_and(data >= 0, data < self.ogm._free_threshold)
        h, w = free.shape
        min_gx = margin_cells
        max_gx = w - 1 - margin_cells
        min_gy = margin_cells
        max_gy = h - 1 - margin_cells
        if min_gx > max_gx or min_gy > max_gy:
            return []

        waypoints: List[Tuple[float, float]] = []
        forward = True

        if sweep_axis == 'x':
            line_indices = range(min_gy, max_gy + 1, step)
            for gy in line_indices:
                segs = self._segments_on_line(free[gy, :], min_len_cells)
                if not segs:
                    continue
                segs.sort(key=lambda s: s[0], reverse=not forward)
                for start, end in segs:
                    start = max(start, min_gx)
                    end = min(end, max_gx)
                    if end < start or (end - start + 1) < min_len_cells:
                        continue
                    a = start + 1 if end - start >= 2 else start
                    b = end - 1 if end - start >= 2 else end
                    if forward:
                        p1 = self.ogm.grid_to_world(a, gy)
                        p2 = self.ogm.grid_to_world(b, gy)
                    else:
                        p1 = self.ogm.grid_to_world(b, gy)
                        p2 = self.ogm.grid_to_world(a, gy)
                    waypoints.append(p1)
                    waypoints.append(p2)
                forward = not forward
        else:
            line_indices = range(min_gx, max_gx + 1, step)
            for gx in line_indices:
                segs = self._segments_on_line(free[:, gx], min_len_cells)
                if not segs:
                    continue
                segs.sort(key=lambda s: s[0], reverse=not forward)
                for start, end in segs:
                    start = max(start, min_gy)
                    end = min(end, max_gy)
                    if end < start or (end - start + 1) < min_len_cells:
                        continue
                    a = start + 1 if end - start >= 2 else start
                    b = end - 1 if end - start >= 2 else end
                    if forward:
                        p1 = self.ogm.grid_to_world(gx, a)
                        p2 = self.ogm.grid_to_world(gx, b)
                    else:
                        p1 = self.ogm.grid_to_world(gx, b)
                        p2 = self.ogm.grid_to_world(gx, a)
                    waypoints.append(p1)
                    waypoints.append(p2)
                forward = not forward

        return self._dedupe_nearby(waypoints, threshold=max(0.05, 0.25 * self.lap_spacing_m))

    @staticmethod
    def _segments_on_line(line: np.ndarray, min_len_cells: int) -> List[Tuple[int, int]]:
        segs: List[Tuple[int, int]] = []
        i = 0
        n = int(line.shape[0])
        while i < n:
            while i < n and not bool(line[i]):
                i += 1
            if i >= n:
                break
            s = i
            while i < n and bool(line[i]):
                i += 1
            e = i - 1
            if (e - s + 1) >= min_len_cells:
                segs.append((s, e))
        return segs

    @staticmethod
    def _dedupe_nearby(pts: List[Tuple[float, float]], threshold: float) -> List[Tuple[float, float]]:
        if not pts:
            return pts
        out = [pts[0]]
        t2 = threshold * threshold
        for x, y in pts[1:]:
            px, py = out[-1]
            if (x - px) * (x - px) + (y - py) * (y - py) <= t2:
                continue
            out.append((x, y))
        return out
