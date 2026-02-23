"""Generate progressive frontier samples and keep sampling connectivity."""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from scipy.ndimage import distance_transform_edt

from .utils import (
    grid_to_world, find_contiguous_segments,
    is_collision_free, euclidean_distance,
)
from .map_manager import MapManager, OBSTACLE_THRESHOLD


@dataclass
class FrontierSample:
    """A frontier sample in world coordinates placed on a lap."""
    x: float
    y: float
    grid_row: int
    grid_col: int
    lap_index: int         # unique segment-lap id
    base_lap_index: int    # geometric lap k (before obstacle segment split)
    position_on_lap: int   # row index on the lap (for sorting along Y)


class ProgressiveSampler:

    def __init__(
        self,
        w: float = 0.30,
        delta: int = 1,
        sweep_direction: str = "x",
        obstacle_buffer: float = 0.12,
        frontier_min_cells: int = 3,
    ):
        self.w = w
        self.delta = delta
        self.sweep_direction = sweep_direction
        self.obstacle_buffer = obstacle_buffer
        self.frontier_min_cells = frontier_min_cells

        self.start_x: Optional[float] = None
        self.start_y: Optional[float] = None

        self._empty_front_streak: int = 0

    @staticmethod
    def _compose_segment_lap_index(base_k: int, segment_idx: int) -> int:
        """Compose a deterministic unique lap id for one segment on base lap k."""
        return base_k * 1_000_000 + segment_idx

    def set_start(self, x: float, y: float):
        """Set anchor position (robot start).  Lap 0 passes through x."""
        self.start_x = x
        self.start_y = y

    def progressive_sample(self, map_manager: MapManager) -> List[FrontierSample]:
        if self.start_x is None:
            return []

        sampling_front = map_manager.get_sampling_front_filtered()
        if not np.any(sampling_front):
            return []

        laps = self._get_lap_positions(sampling_front, map_manager)
        if not laps:
            map_manager.mark_sampling_front_done(sampling_front)
            return []

        all_samples: List[FrontierSample] = []
        spacing_cells = max(1, int(self.delta * self.w / map_manager.resolution))

        for lap_index, lap_grid_pos in laps:
            segment_counter = 0
            if self.sweep_direction == "x":
                grid_col = lap_grid_pos
                if grid_col < 0 or grid_col >= map_manager.grid_width:
                    continue
                line = sampling_front[:, grid_col]
            else:
                grid_row_fixed = lap_grid_pos
                if grid_row_fixed < 0 or grid_row_fixed >= map_manager.grid_height:
                    continue
                line = sampling_front[grid_row_fixed, :]

            segments = find_contiguous_segments(line)

            for seg_start, seg_end in segments:
                segment_counter += 1
                seg_lap_index = self._compose_segment_lap_index(
                    lap_index, segment_counter
                )
                pos = seg_start
                last_placed = seg_start - spacing_cells  # allow first

                while pos <= seg_end:
                    if self.sweep_direction == "x":
                        row, col = pos, lap_grid_pos
                    else:
                        row, col = lap_grid_pos, pos

                    if not map_manager.is_free(row, col):
                        pos += 1
                        continue

                    if map_manager.is_already_covered(row, col):
                        pos += 1
                        continue

                    if not map_manager.is_safe_from_obstacles(row, col):
                        pos += 1
                        continue

                    wx, wy = grid_to_world(
                        row, col,
                        map_manager.origin_x, map_manager.origin_y,
                        map_manager.resolution,
                    )
                    all_samples.append(FrontierSample(
                        x=wx, y=wy,
                        grid_row=row, grid_col=col,
                        lap_index=seg_lap_index,
                        base_lap_index=lap_index,
                        position_on_lap=pos,
                    ))
                    last_placed = pos
                    pos += spacing_cells   # jump forward after placement

        wall_samples = self._generate_wall_adjacent_samples(
            sampling_front, map_manager, all_samples, spacing_cells,
        )
        if wall_samples:
            all_samples.extend(wall_samples)

        all_samples = self._ensure_connectivity(
            all_samples, map_manager, spacing_cells,
        )

        if all_samples:
            self._empty_front_streak = 0
            coverage_mask = self._build_coverage_mask(
                all_samples, map_manager, sampling_front,
            )
            map_manager.mark_sampling_front_done(coverage_mask)
        else:
            self._empty_front_streak += 1
            if self._empty_front_streak >= 2:
                map_manager.mark_sampling_front_done(sampling_front)
                self._empty_front_streak = 0

        return all_samples

    def _generate_wall_adjacent_samples(
        self,
        sampling_front: np.ndarray,
        map_manager: MapManager,
        existing_samples: List[FrontierSample],
        spacing_cells: int,
    ) -> List[FrontierSample]:
        mm = map_manager
        if self.obstacle_buffer <= 0 or mm.occupancy_grid is None:
            return []

        buffer_cells = int(self.obstacle_buffer / mm.resolution)
        if buffer_cells < 1:
            return []

        h, w_grid = mm.occupancy_grid.shape

        obstacle_mask = mm.occupancy_grid >= OBSTACLE_THRESHOLD
        dist_map = distance_transform_edt(~obstacle_mask)

        lo = max(1, buffer_cells - 1)
        hi = buffer_cells + 2                       # slightly generous outer rim
        at_buffer_band = (dist_map >= lo) & (dist_map <= hi)

        candidates_mask = (
            at_buffer_band
            & sampling_front
            & (mm.occupancy_grid == 0)
        )

        wall_excl = max(1, spacing_cells // 3)   # ≈ 0.25m instead of 0.75m
        for s in existing_samples:
            r0 = max(0, s.grid_row - wall_excl)
            r1 = min(h, s.grid_row + wall_excl + 1)
            c0 = max(0, s.grid_col - wall_excl)
            c1 = min(w_grid, s.grid_col + wall_excl + 1)
            candidates_mask[r0:r1, c0:c1] = False

        if not np.any(candidates_mask):
            return []

        rows, cols = np.where(candidates_mask)
        if self.sweep_direction == "x":
            order = np.lexsort((rows, cols))        # col first, then row
        else:
            order = np.lexsort((cols, rows))        # row first, then col
        rows = rows[order]
        cols = cols[order]

        new_samples: List[FrontierSample] = []
        placed = np.zeros((h, w_grid), dtype=bool)  # mutual spacing guard
        wall_mutual = max(1, spacing_cells // 2)

        for idx in range(len(rows)):
            r, c = int(rows[idx]), int(cols[idx])

            r0 = max(0, r - wall_mutual)
            r1 = min(h, r + wall_mutual + 1)
            c0 = max(0, c - wall_mutual)
            c1 = min(w_grid, c + wall_mutual + 1)
            if placed[r0:r1, c0:c1].any():
                continue

            # Removed filters except obstacle buffer which is already handled
            # by candidates_mask and dist_map constraint earlier.

            wx, wy = grid_to_world(
                r, c, mm.origin_x, mm.origin_y, mm.resolution,
            )
            if self.sweep_direction == "x":
                base_k = round((wx - self.start_x) / self.w)
            else:
                base_k = round((wy - self.start_y) / self.w)

            seg_lap, seg_base = self._nearest_segment_lap_index(
                r, c, existing_samples, base_k,
            )

            new_samples.append(FrontierSample(
                x=wx, y=wy,
                grid_row=r, grid_col=c,
                lap_index=seg_lap,
                base_lap_index=seg_base,
                position_on_lap=r if self.sweep_direction == "x" else c,
            ))
            placed[r0:r1, c0:c1] = True

        return new_samples

    def _nearest_segment_lap_index(
        self,
        row: int,
        col: int,
        samples: List[FrontierSample],
        default_base_k: int,
    ) -> Tuple[int, int]:
        """
        Pick nearest existing segment-lap to (row,col); fallback to base-k seg#0.
        """
        if not samples:
            return self._compose_segment_lap_index(default_base_k, 0), default_base_k

        best = None
        best_d = float("inf")
        for s in samples:
            d = (s.grid_row - row) ** 2 + (s.grid_col - col) ** 2
            if d < best_d:
                best = s
                best_d = d
        if best is None:
            return self._compose_segment_lap_index(default_base_k, 0), default_base_k
        return best.lap_index, best.base_lap_index

    def _ensure_connectivity(
        self,
        samples: List[FrontierSample],
        map_manager: MapManager,
        spacing_cells: int,
    ) -> List[FrontierSample]:
        if len(samples) < 2 or map_manager.occupancy_grid is None:
            return samples

        cross_lap_max = 1.5 * self.w
        max_iterations = 3

        for _iteration in range(max_iterations):
            adj = self._build_sample_adjacency(
                samples, map_manager, cross_lap_max,
            )
            comps = self._sample_connected_components(samples, adj)
            if len(comps) <= 1:
                return samples          # already connected

            anchor = comps[0]
            anchor_set: Set[int] = set(anchor)
            bridge_samples: List[FrontierSample] = []

            for comp in comps[1:]:
                bridges = self._bridge_components(
                    samples, anchor_set, set(comp),
                    map_manager, spacing_cells,
                )
                if bridges:
                    bridge_samples.extend(bridges)
                    anchor_set.update(comp)

            if not bridge_samples:
                break                   # can't bridge → give up
            samples = samples + bridge_samples

        return samples


    def _build_sample_adjacency(
        self,
        samples: List[FrontierSample],
        mm: MapManager,
        cross_lap_max: float,
    ) -> Dict[int, Set[int]]:
        """Return adjacency dict (sample-index → set of neighbor indices)."""
        n = len(samples)
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

        laps: Dict[int, List[int]] = {}
        for i, s in enumerate(samples):
            laps.setdefault(s.lap_index, []).append(i)

        for lap_idx, idxs in laps.items():
            idxs.sort(key=lambda i: samples[i].position_on_lap)
            for j in range(len(idxs) - 1):
                a, b = idxs[j], idxs[j + 1]
                sa, sb = samples[a], samples[b]
                if is_collision_free(
                    sa.grid_row, sa.grid_col,
                    sb.grid_row, sb.grid_col,
                    mm.occupancy_grid,
                ):
                    adj[a].add(b)
                    adj[b].add(a)

        lap_keys = sorted(laps.keys())
        lap_base: Dict[int, int] = {}
        for lap_idx, idxs in laps.items():
            if idxs:
                lap_base[lap_idx] = samples[idxs[0]].base_lap_index
        for ki, k in enumerate(lap_keys):
            for kk in lap_keys[ki + 1:]:
                bk = lap_base.get(k, k)
                bkk = lap_base.get(kk, kk)
                if bk == bkk:
                    continue
                if abs(bk - bkk) > 1:
                    continue
                for ai in laps[k]:
                    sa = samples[ai]
                    for bi in laps[kk]:
                        sb = samples[bi]
                        d = euclidean_distance(sa.x, sa.y, sb.x, sb.y)
                        if d > cross_lap_max:
                            continue
                        if is_collision_free(
                            sa.grid_row, sa.grid_col,
                            sb.grid_row, sb.grid_col,
                            mm.occupancy_grid,
                        ):
                            adj[ai].add(bi)
                            adj[bi].add(ai)

        return adj

    @staticmethod
    def _sample_connected_components(
        samples: List[FrontierSample],
        adj: Dict[int, Set[int]],
    ) -> List[List[int]]:
        """Connected components as lists of sample indices."""
        n = len(samples)
        visited: Set[int] = set()
        comps: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            stack = [start]
            comp: List[int] = []
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                comp.append(idx)
                for nb in adj.get(idx, set()):
                    if nb not in visited:
                        stack.append(nb)
            comps.append(comp)
        return comps

    def _bridge_components(
        self,
        samples: List[FrontierSample],
        comp_a: Set[int],
        comp_b: Set[int],
        mm: MapManager,
        spacing_cells: int,
    ) -> List[FrontierSample]:
        best_d = float('inf')
        best_pair: Optional[Tuple[int, int]] = None
        for ai in comp_a:
            sa = samples[ai]
            for bi in comp_b:
                sb = samples[bi]
                d = euclidean_distance(sa.x, sa.y, sb.x, sb.y)
                if d < best_d:
                    best_d = d
                    best_pair = (ai, bi)

        if best_pair is None:
            return []

        sa = samples[best_pair[0]]
        sb = samples[best_pair[1]]

        max_path_cells = int(30.0 * self.w / mm.resolution)

        path_cells = self._bfs_free_path(
            mm.occupancy_grid,
            sa.grid_row, sa.grid_col,
            sb.grid_row, sb.grid_col,
            max_cells=max_path_cells,
        )
        if not path_cells:
            return []

        bridge: List[FrontierSample] = []
        step = max(1, spacing_cells)
        for i in range(step, len(path_cells) - step, step):
            r, c = path_cells[i]
            if not mm.is_free(r, c):
                continue
            wx, wy = grid_to_world(
                r, c, mm.origin_x, mm.origin_y, mm.resolution,
            )
            if self.sweep_direction == "x":
                base_k = round((wx - self.start_x) / self.w)
            else:
                base_k = round((wy - self.start_y) / self.w)
            seg_lap, seg_base = self._nearest_segment_lap_index(
                r, c, samples, base_k,
            )

            bridge.append(FrontierSample(
                x=wx, y=wy,
                grid_row=r, grid_col=c,
                lap_index=seg_lap,
                base_lap_index=seg_base,
                position_on_lap=r if self.sweep_direction == "x" else c,
            ))

        return bridge

    @staticmethod
    def _bfs_free_path(
        grid: np.ndarray,
        r0: int, c0: int,
        r1: int, c1: int,
        max_cells: int = 500,
    ) -> List[Tuple[int, int]]:
        h, w = grid.shape
        if not (0 <= r0 < h and 0 <= c0 < w and 0 <= r1 < h and 0 <= c1 < w):
            return []

        visited = np.zeros((h, w), dtype=bool)
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
        queue: deque = deque()
        queue.append((r0, c0))
        visited[r0, c0] = True
        explored = 0

        while queue:
            r, c = queue.popleft()
            if r == r1 and c == c1:
                path: List[Tuple[int, int]] = []
                cur = (r1, c1)
                while cur != (r0, c0):
                    path.append(cur)
                    cur = prev[cur]
                path.append((r0, c0))
                path.reverse()
                return path

            explored += 1
            if explored > max_cells:
                return []   # too far apart

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if int(grid[nr, nc]) == 0:  # FREE only
                        visited[nr, nc] = True
                        prev[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        return []   # no path

    def _build_coverage_mask(
        self,
        samples: List[FrontierSample],
        mm: MapManager,
        sampling_front: np.ndarray,
    ) -> np.ndarray:
        h, w_grid = sampling_front.shape
        mask = np.zeros((h, w_grid), dtype=bool)
        cov_r = max(1, int(self.w / (2.0 * mm.resolution)))  # w/2

        for s in samples:
            r0 = max(0, s.grid_row - cov_r)
            r1 = min(h, s.grid_row + cov_r + 1)
            c0 = max(0, s.grid_col - cov_r)
            c1 = min(w_grid, s.grid_col + cov_r + 1)
            mask[r0:r1, c0:c1] = True

        return sampling_front & mask

    def _get_lap_positions(
        self,
        sampling_front: np.ndarray,
        map_manager: MapManager
    ) -> List[tuple]:
        """Return list of (lap_index, grid_col) that intersect the front."""
        if self.sweep_direction == "x":
            return self._get_lap_positions_x(sampling_front, map_manager)
        else:
            return self._get_lap_positions_y(sampling_front, map_manager)

    def _get_lap_positions_x(self, sampling_front, mm: MapManager):
        """Vertical laps parallel to Y-axis."""
        cols_with_front = np.any(sampling_front, axis=0)
        if not np.any(cols_with_front):
            return []

        active_cols = np.where(cols_with_front)[0]
        min_x = mm.origin_x + active_cols[0] * mm.resolution
        max_x = mm.origin_x + active_cols[-1] * mm.resolution

        k_min = math.floor((min_x - self.start_x) / self.w)
        k_max = math.ceil((max_x - self.start_x) / self.w)

        laps = []
        for k in range(k_min, k_max + 1):
            lap_x = self.start_x + k * self.w
            grid_col = int((lap_x - mm.origin_x) / mm.resolution)
            if 0 <= grid_col < mm.grid_width:
                if np.any(sampling_front[:, grid_col]):
                    laps.append((k, grid_col))
        return laps

    def _get_lap_positions_y(self, sampling_front, mm: MapManager):
        """Horizontal laps parallel to X-axis (mirror logic)."""
        rows_with_front = np.any(sampling_front, axis=1)
        if not np.any(rows_with_front):
            return []

        active_rows = np.where(rows_with_front)[0]
        min_y = mm.origin_y + active_rows[0] * mm.resolution
        max_y = mm.origin_y + active_rows[-1] * mm.resolution

        k_min = math.floor((min_y - self.start_y) / self.w)
        k_max = math.ceil((max_y - self.start_y) / self.w)

        laps = []
        for k in range(k_min, k_max + 1):
            lap_y = self.start_y + k * self.w
            grid_row = int((lap_y - mm.origin_y) / mm.resolution)
            if 0 <= grid_row < mm.grid_height:
                if np.any(sampling_front[grid_row, :]):
                    laps.append((k, grid_row))
        return laps
