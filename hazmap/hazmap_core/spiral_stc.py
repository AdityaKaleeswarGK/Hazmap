"""Known-map coverage planning with a Spiral-STC style tree traversal."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


Grid = Tuple[int, int]  # (row, col)


class SpiralSTCPlanner:
    """Build a coarse free-space graph and return a tree-walk coverage route."""

    def __init__(self, ogm, cell_size_m: float = 0.5):
        self.ogm = ogm
        self.cell_size_m = max(0.1, float(cell_size_m))

    def plan(self, start_x: float, start_y: float) -> List[Tuple[float, float]]:
        if not self.ogm.ready:
            return []
        if self.ogm._data is None:
            return []

        coarse_step = max(1, int(round(self.cell_size_m / self.ogm.resolution)))
        coarse_free = self._build_coarse_free_mask(coarse_step)
        if coarse_free.size == 0 or not np.any(coarse_free):
            return []

        start_rc = self._start_coarse_cell(start_x, start_y, coarse_step, coarse_free)
        if start_rc is None:
            return []

        parent, children = self._build_dfs_tree(start_rc, coarse_free)
        order = self._tree_walk_order(start_rc, children)
        if not order:
            return []

        waypoints: List[Tuple[float, float]] = []
        for cr, cc in order:
            wx, wy = self._coarse_to_world(cr, cc, coarse_step)
            waypoints.append((wx, wy))
        return waypoints

    def _build_coarse_free_mask(self, step: int) -> np.ndarray:
        data = self.ogm._data
        h, w = data.shape
        ch = h // step
        cw = w // step
        if ch == 0 or cw == 0:
            return np.zeros((0, 0), dtype=bool)

        coarse = np.zeros((ch, cw), dtype=bool)
        free_thresh = self.ogm._free_threshold
        for cr in range(ch):
            r0 = cr * step
            r1 = min(h, r0 + step)
            for cc in range(cw):
                c0 = cc * step
                c1 = min(w, c0 + step)
                block = data[r0:r1, c0:c1]
                # Conservative: require at least 80% free (known, non-obstacle)
                known_free = np.logical_and(block >= 0, block < free_thresh)
                if known_free.size > 0 and (np.sum(known_free) / known_free.size) >= 0.8:
                    coarse[cr, cc] = True
        return coarse

    def _start_coarse_cell(
        self,
        start_x: float,
        start_y: float,
        step: int,
        coarse_free: np.ndarray,
    ) -> Optional[Grid]:
        gx, gy = self.ogm.world_to_grid(start_x, start_y)  # (col,row)
        cr = gy // step
        cc = gx // step
        h, w = coarse_free.shape
        if 0 <= cr < h and 0 <= cc < w and coarse_free[cr, cc]:
            return (cr, cc)

        free_cells = np.argwhere(coarse_free)
        if free_cells.size == 0:
            return None
        d2 = (free_cells[:, 0] - cr) ** 2 + (free_cells[:, 1] - cc) ** 2
        idx = int(np.argmin(d2))
        return (int(free_cells[idx, 0]), int(free_cells[idx, 1]))

    def _neighbors(self, node: Grid, coarse_free: np.ndarray) -> List[Grid]:
        r, c = node
        out: List[Grid] = []
        h, w = coarse_free.shape
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and coarse_free[nr, nc]:
                out.append((nr, nc))
        return out

    def _build_dfs_tree(
        self, start: Grid, coarse_free: np.ndarray
    ) -> Tuple[Dict[Grid, Optional[Grid]], Dict[Grid, List[Grid]]]:
        parent: Dict[Grid, Optional[Grid]] = {start: None}
        children: Dict[Grid, List[Grid]] = {}
        visited: Set[Grid] = set([start])
        stack: List[Grid] = [start]

        while stack:
            u = stack.pop()
            nbrs = self._neighbors(u, coarse_free)
            # Spiral-ish bias: deterministic clockwise preference by row/col parity
            nbrs.sort(key=lambda x: (x[0], x[1]))
            for v in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                parent[v] = u
                children.setdefault(u, []).append(v)
                stack.append(v)
        return parent, children

    def _tree_walk_order(self, root: Grid, children: Dict[Grid, List[Grid]]) -> List[Grid]:
        order: List[Grid] = []

        def dfs(u: Grid):
            order.append(u)
            for v in children.get(u, []):
                dfs(v)
                order.append(u)  # backtrack step

        dfs(root)

        compact: List[Grid] = []
        for n in order:
            if not compact or compact[-1] != n:
                compact.append(n)
        return compact

    def _coarse_to_world(self, cr: int, cc: int, step: int) -> Tuple[float, float]:
        """Pick a representative free point within the coarse cell.

        Using the exact block center can place waypoints on an occupied/unknown
        pixel near obstacles. We instead snap to the nearest known-free cell in
        the block, falling back to center only if none exists.
        """
        data = self.ogm._data
        h, w = data.shape
        r0 = cr * step
        r1 = min(h, r0 + step)
        c0 = cc * step
        c1 = min(w, c0 + step)

        cgy = r0 + (r1 - r0) // 2
        cgx = c0 + (c1 - c0) // 2

        best = None
        best_d2 = float('inf')
        free_thresh = self.ogm._free_threshold
        for gy in range(r0, r1):
            for gx in range(c0, c1):
                v = data[gy, gx]
                if v < 0 or v >= free_thresh:
                    continue
                d2 = (gy - cgy) ** 2 + (gx - cgx) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (gx, gy)

        if best is None:
            gx, gy = cgx, cgy
        else:
            gx, gy = best

        wx, wy = self.ogm.grid_to_world(gx, gy)
        return (wx, wy)
