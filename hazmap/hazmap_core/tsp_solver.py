"""
TSP-based Coverage Hole Prevention for the C* algorithm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .rcg import RCG, NodeState


@dataclass
class TSPWaypoint:
    x: float
    y: float
    node_id: Optional[int] = None


@dataclass
class TSPPlan:
    waypoints: List[TSPWaypoint]
    hole_node_ids: Set[int]
    ends_at_goal: bool


class TSPSolver:
    def __init__(self, rcg: RCG):
        self.rcg = rcg

    def detect_coverage_holes(self, current_id: int, goal_id: int) -> List[Set[int]]:
        current = self.rcg.nodes.get(current_id)
        if current is None:
            return []

        labeled: Set[int] = set()
        holes: List[Set[int]] = []
        for nb_id in current.all_neighbor_ids():
            nb = self.rcg.nodes.get(nb_id)
            if nb is None or nb.state != NodeState.OPEN or nb_id == goal_id or nb_id in labeled:
                continue

            component = self._flood_open_component(nb_id, goal_id)
            labeled.update(component)
            if component and not self._touches_unknown(component):
                holes.append(component)
        return holes

    def compute_tsp_plan(self, hole_nodes: Set[int], current_id: int, goal_id: int) -> Optional[TSPPlan]:
        current = self.rcg.nodes.get(current_id)
        goal = self.rcg.nodes.get(goal_id)
        if current is None or goal is None or not hole_nodes:
            return None

        region_cells = self._discover_hole_region(hole_nodes)
        appended = self._append_nodes_from_region(region_cells)

        points: List[TSPWaypoint] = [
            TSPWaypoint(current.x, current.y, current.id),
            TSPWaypoint(goal.x, goal.y, goal.id),
        ]
        for nid in sorted(hole_nodes):
            node = self.rcg.nodes.get(nid)
            if node is not None:
                points.append(TSPWaypoint(node.x, node.y, nid))
        points.extend(appended)
        points = self._dedupe_points(points)
        if len(points) <= 2:
            return None

        start_idx = self._find_idx(points, current.id)
        goal_idx = self._find_idx(points, goal.id)

        # Algorithm 3, conditions 1-3 for n_end^TSP selection
        end_condition = self._tsp_end_condition(
            current_id, goal_id, hole_nodes)

        if end_condition == 1:
            # Condition 1: end at goal node
            order = self._solve_open_tsp(points, start_idx, goal_idx)
            return TSPPlan([points[i] for i in order[1:]], set(hole_nodes), True)
        elif end_condition == 2:
            # Condition 2: end at current node (return after covering hole)
            order = self._solve_open_tsp(points, start_idx, start_idx)
            return TSPPlan([points[i] for i in order[1:]], set(hole_nodes), False)
        else:
            # Condition 3: unspecified → cycle TSP
            order = self._solve_cycle_tsp(points, start_idx)
            return TSPPlan([points[i] for i in order[1:]], set(hole_nodes), False)

    def _flood_open_component(self, start_id: int, goal_id: int) -> Set[int]:
        comp: Set[int] = set()
        queue: deque[int] = deque([start_id])
        while queue:
            nid = queue.popleft()
            if nid in comp or nid == goal_id:
                continue
            node = self.rcg.nodes.get(nid)
            if node is None or node.state != NodeState.OPEN:
                continue
            comp.add(nid)
            for next_id in node.all_neighbor_ids():
                if next_id not in comp and next_id != goal_id:
                    queue.append(next_id)
        return comp

    def _touches_unknown(self, comp: Set[int]) -> bool:
        for nid in comp:
            node = self.rcg.nodes.get(nid)
            if node is not None and self.rcg.ogm.is_adjacent_to_unknown(node.x, node.y, self.rcg.w):
                return True
        return False

    def _discover_hole_region(self, hole_nodes: Set[int]) -> Set[Tuple[int, int]]:
        ogm = self.rcg.ogm
        if not ogm.ready or ogm._data is None:
            return set()

        seeds: List[Tuple[int, int]] = []
        for nid in hole_nodes:
            node = self.rcg.nodes.get(nid)
            if node is None:
                continue
            gx, gy = ogm.world_to_grid(node.x, node.y)
            seeds.append((gx, gy))
        if not seeds:
            return set()

        xs = [gx for gx, _ in seeds]
        ys = [gy for _, gy in seeds]
        pad = max(2, int(math.ceil((2.0 * self.rcg.w) / ogm.resolution)))
        min_x = max(0, min(xs) - pad)
        max_x = min(ogm._width - 1, max(xs) + pad)
        min_y = max(0, min(ys) - pad)
        max_y = min(ogm._height - 1, max(ys) + pad)

        region: Set[Tuple[int, int]] = set()
        visited: Set[Tuple[int, int]] = set(seeds)
        queue: deque[Tuple[int, int]] = deque(seeds)

        while queue:
            gx, gy = queue.popleft()
            if gx < min_x or gx > max_x or gy < min_y or gy > max_y:
                continue
            if not ogm._in_bounds(gx, gy):
                continue
            val = ogm._cell_value(gx, gy)
            if val < 0 or val >= ogm._free_threshold:
                continue
            wx, wy = ogm.grid_to_world(gx, gy)
            if ogm.is_adjacent_to_unknown(wx, wy, ogm.resolution):
                continue

            region.add((gx, gy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (gx + dx, gy + dy)
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return region

    def _append_nodes_from_region(self, region: Set[Tuple[int, int]]) -> List[TSPWaypoint]:
        if not region:
            return []
        ogm = self.rcg.ogm
        spacing_cells = max(1, int(round(self.rcg.w / ogm.resolution)))
        cols = sorted({gx for gx, _ in region})
        by_col: Dict[int, List[int]] = {}
        for gx, gy in region:
            by_col.setdefault(gx, []).append(gy)

        waypoints: List[TSPWaypoint] = []
        for gx in range(cols[0], cols[-1] + 1, spacing_cells):
            ys = sorted(by_col.get(gx, []))
            if not ys:
                continue
            seg_start = ys[0]
            prev = ys[0]
            for gy in ys[1:] + [None]:
                if gy is not None and gy == prev + 1:
                    prev = gy
                    continue
                pos = seg_start
                while pos <= prev:
                    wx, wy = ogm.grid_to_world(gx, pos)
                    waypoints.append(TSPWaypoint(wx, wy))
                    pos += spacing_cells
                if gy is None:
                    break
                seg_start = gy
                prev = gy
        return waypoints

    def _dedupe_points(self, points: List[TSPWaypoint]) -> List[TSPWaypoint]:
        """
        Remove waypoints that land in the same grid cell.
        Prefer waypoints that have a node_id over generic waypoints.
        """
        # Map (gx, gy) -> TSPWaypoint
        best_points: Dict[Tuple[int, int], TSPWaypoint] = {}
        for p in points:
            gx, gy = self.rcg.ogm.world_to_grid(p.x, p.y)
            key = (gx, gy)
            
            if key not in best_points:
                best_points[key] = p
            else:
                # If we already have a point here, only replace it if 
                # the new one has a node_id and the old one doesn't.
                if p.node_id is not None and best_points[key].node_id is None:
                    best_points[key] = p
        
        return list(best_points.values())

    @staticmethod
    def _find_idx(points: List[TSPWaypoint], node_id: int) -> int:
        for i, p in enumerate(points):
            if p.node_id == node_id:
                return i
        raise ValueError(node_id)

    def _tsp_end_condition(self, current_id: int, goal_id: int,
                           hole_nodes: Set[int]) -> int:
        """Algorithm 3, Lines 3-9: determine TSP end-node condition.

        Returns 1, 2, or 3 corresponding to the paper's conditions.
        """
        goal = self.rcg.nodes.get(goal_id)
        current = self.rcg.nodes.get(current_id)

        # Condition 1: n_{i+1} has an Open neighbor not in N_CH,
        #              or n_{i+1} is adjacent to unknown
        if goal is not None:
            if self.rcg.ogm.is_adjacent_to_unknown(goal.x, goal.y, self.rcg.w):
                return 1
            for nb_id in goal.all_neighbor_ids():
                nb = self.rcg.nodes.get(nb_id)
                if nb and nb.state == NodeState.OPEN and nb_id not in hole_nodes:
                    return 1

        # Condition 2: n_i has an Open neighbor not in N_CH
        if current is not None:
            for nb_id in current.all_neighbor_ids():
                nb = self.rcg.nodes.get(nb_id)
                if nb and nb.state == NodeState.OPEN and nb_id not in hole_nodes:
                    return 2

        # Condition 3: unspecified
        return 3

    def _solve_open_tsp(self, points: List[TSPWaypoint], start_idx: int, end_idx: int) -> List[int]:
        dist = self._distance_matrix(points)
        mids = [i for i in range(len(points)) if i not in (start_idx, end_idx)]
        route = [start_idx]
        remaining = set(mids)
        cur = start_idx
        while remaining:
            nxt = min(remaining, key=lambda i: dist[cur][i])
            route.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        route.append(end_idx)
        return self._two_opt_open(route, dist)

    def _solve_cycle_tsp(self, points: List[TSPWaypoint], start_idx: int) -> List[int]:
        dist = self._distance_matrix(points)
        route = [start_idx]
        remaining = set(i for i in range(len(points)) if i != start_idx)
        cur = start_idx
        while remaining:
            nxt = min(remaining, key=lambda i: dist[cur][i])
            route.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        route = self._two_opt_cycle(route, dist)
        route.append(start_idx)
        return route

    @staticmethod
    def _two_opt_open(route: List[int], dist: List[List[float]]) -> List[int]:
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    a, b = route[i - 1], route[i]
                    c, d = route[j], route[j + 1]
                    if dist[a][c] + dist[b][d] + 1e-9 < dist[a][b] + dist[c][d]:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True
        return route

    @staticmethod
    def _two_opt_cycle(route: List[int], dist: List[List[float]]) -> List[int]:
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    a, b = route[i - 1], route[i]
                    c = route[j]
                    d = route[(j + 1) % n]
                    if dist[a][c] + dist[b][d] + 1e-9 < dist[a][b] + dist[c][d]:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True
        return route

    def _distance_matrix(self, points: List[TSPWaypoint]) -> List[List[float]]:
        """Euclidean distance matrix — fast O(N²) for TSP ordering.

        Coverage holes are in free space so Euclidean distance is a good
        approximation.  The old grid-A* version was O(N² × grid_cells)
        and blocked the coverage thread for seconds on large holes.
        """
        n = len(points)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(points[i].x - points[j].x,
                               points[i].y - points[j].y)
                dist[i][j] = d
                dist[j][i] = d
        return dist
