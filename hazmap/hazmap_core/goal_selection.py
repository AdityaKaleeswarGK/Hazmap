"""
Goal Node Selection and State Update for the C* algorithm.

Implements:
  • Algorithm 1 – SelectGoalNode (priority: Left → Up → Down → Right)
  • Algorithm 2 – UpdateState (mark Closed, create link nodes)
  • Dead-end escape via retreat nodes + A* on RCG
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .rcg import RCG, NodeState, RCGNode


class GoalSelector:
    """Handles goal selection, state updates, and dead-end escape."""

    def __init__(self, rcg: RCG):
        self.rcg = rcg
        self.retreat_nodes: Set[int] = set()

    def select_goal_node(self, current_id: int) -> Optional[int]:
        node = self.rcg.nodes.get(current_id)
        if node is None:
            return None

        for direction in ["left", "up", "down", "right"]:
            nb = self.rcg.get_open_neighbor(node, direction)
            if nb is not None:
                return nb.id

        return self._escape_dead_end(current_id)

    def update_state(self, current_id: int, goal_id: int) -> None:
        """Algorithm 2: Update the state of the current node.

        Paper notation:
          n_i^L  = Left cross-lap neighbour
          n_i^U  = Up (sweep-forward) same-lap neighbour
          n_i^P  = Previous same-lap neighbour (sweep-backward = 'down')

        Paper Line 2: Close n_i when BOTH Up and Down neighbours are Closed
        (the lap segment through n_i has been fully covered in both directions).
        """
        node = self.rcg.nodes.get(current_id)
        if node is None:
            return

        # Paper Alg.2 line 2: close n_i if EITHER Up OR Down neighbour is Closed.
        # Keep Open only when BOTH neighbours are still Open, to preserve the
        # back-and-forth pattern through interior lap nodes.
        up_nb = self.rcg.get_neighbor(node, "up")
        down_nb = self.rcg.get_neighbor(node, "down")

        up_closed = up_nb is not None and up_nb.state == NodeState.CLOSED
        down_closed = down_nb is not None and down_nb.state == NodeState.CLOSED

        up_absent = up_nb is None and node.is_end_node
        down_absent = down_nb is None and node.is_end_node

        if up_closed or down_closed or up_absent or down_absent:
            self.rcg.set_node_state(current_id, NodeState.CLOSED)

        # ── Paper Lines 5-11: create link nodes when goal is Left or Previous ──
        goal_node = self.rcg.nodes.get(goal_id)
        if goal_node is None or node.state != NodeState.CLOSED:
            return

        goal_on_left = goal_id in node.neighbors.get("left", [])
        goal_is_prev = goal_id in node.neighbors.get("down", [])

        # Paper Line 5-8: if n_{i+1} = n_i^L and q(n_i^U) = Op → link above
        if goal_on_left:
            if up_nb and up_nb.state == NodeState.OPEN:
                dist = math.hypot(node.x - up_nb.x, node.y - up_nb.y)
                if dist > self.rcg.w:
                    self._create_link_node(node, up_nb, "up")

        # Paper Line 9-10: if n_{i+1} = n_i^P and q(n_i^P) = Op → link below
        if goal_is_prev:
            if down_nb and down_nb.state == NodeState.OPEN:
                dist = math.hypot(node.x - down_nb.x, node.y - down_nb.y)
                if dist > self.rcg.w:
                    self._create_link_node(node, down_nb, "down")

    def _create_link_node(self, node: RCGNode, target: RCGNode, direction: str) -> None:
        if direction == "up":
            lx, ly = node.x, node.y + self.rcg.w
            lap_position = node.lap_position + self.rcg.w
            new_dir = "up"
            old_dir = "down"
        else:
            lx, ly = node.x, node.y - self.rcg.w
            lap_position = node.lap_position - self.rcg.w
            new_dir = "down"
            old_dir = "up"

        # Reuse any existing node at this position to avoid duplicate waypoints
        tol = 0.05 * self.rcg.w
        existing = next(
            (
                n
                for n in self.rcg.nodes.values()
                if abs(n.x - lx) < tol and abs(n.y - ly) < tol
            ),
            None,
        )
        if existing is not None:
            # Just wire edges/neighbours to existing node and return
            self.rcg.add_edge(existing.id, node.id)
            self.rcg.add_edge(existing.id, target.id)
            if node.id not in existing.neighbors[old_dir]:
                existing.neighbors[old_dir].append(node.id)
            if target.id not in existing.neighbors[new_dir]:
                existing.neighbors[new_dir].append(target.id)
            if existing.id not in node.neighbors[new_dir]:
                node.neighbors[new_dir].append(existing.id)
            if existing.id not in target.neighbors[old_dir]:
                target.neighbors[old_dir].append(existing.id)
            return

        link = self.rcg.add_node(lx, ly, node.lap_index, lap_position, is_link=True)
        self.rcg.add_edge(link.id, node.id)
        self.rcg.add_edge(link.id, target.id)
        if node.id not in link.neighbors[old_dir]:
            link.neighbors[old_dir].append(node.id)
        if target.id not in link.neighbors[new_dir]:
            link.neighbors[new_dir].append(target.id)
        if link.id not in node.neighbors[new_dir]:
            node.neighbors[new_dir].append(link.id)
        if link.id not in target.neighbors[old_dir]:
            target.neighbors[old_dir].append(link.id)

    def update_retreat_nodes(self, robot_x: float, robot_y: float) -> None:
        threshold = math.sqrt(2) * self.rcg.w
        threshold_sq = threshold * threshold

        new_retreat = set()
        for nid in self.rcg._open_ids:
            node = self.rcg.nodes.get(nid)
            if node is None:
                continue
            dx = node.x - robot_x
            dy = node.y - robot_y
            if dx * dx + dy * dy <= threshold_sq:
                new_retreat.add(nid)

        self.retreat_nodes = new_retreat

    def _escape_dead_end(self, current_id: int) -> Optional[int]:
        if not self.retreat_nodes:
            return self._nearest_open_node(current_id)

        targets = set(self.retreat_nodes)
        result = self._multi_target_dijkstra(current_id, targets)
        if result is not None:
            return result

        return self._nearest_open_node(current_id)

    def _multi_target_dijkstra(self, start_id: int, targets: Set[int]) -> Optional[int]:
        """Single Dijkstra search to find nearest target among multiple goals."""
        if start_id in targets:
            return start_id

        import heapq

        dist: Dict[int, float] = {start_id: 0.0}
        heap: List[Tuple[float, int]] = [(0.0, start_id)]
        came_from: Dict[int, int] = {}

        while heap:
            d, current = heapq.heappop(heap)
            if current in targets:
                return current
            if d > dist.get(current, float("inf")):
                continue

            node = self.rcg.nodes.get(current)
            if node is None:
                continue

            for nb_id in node.all_neighbor_ids():
                if nb_id not in self.rcg.nodes:
                    continue
                nb = self.rcg.nodes[nb_id]
                nd = d + math.hypot(node.x - nb.x, node.y - nb.y)
                if nd < dist.get(nb_id, float("inf")):
                    dist[nb_id] = nd
                    came_from[nb_id] = current
                    heapq.heappush(heap, (nd, nb_id))

        return None

    def _nearest_open_node(self, current_id: int) -> Optional[int]:
        """Find the nearest open node by Euclidean distance."""
        if not self.rcg._open_ids:
            return None
        node = self.rcg.nodes.get(current_id)
        if node is None:
            return None
        best_id = None
        best_dist = float("inf")
        for nid in self.rcg._open_ids:
            n = self.rcg.nodes.get(nid)
            if n is None:
                continue
            d = math.hypot(n.x - node.x, n.y - node.y)
            if d < best_dist:
                best_dist = d
                best_id = nid
        return best_id

    def _path_length(self, path_ids: List[int]) -> float:
        total = 0.0
        for a, b in zip(path_ids, path_ids[1:]):
            na = self.rcg.nodes[a]
            nb = self.rcg.nodes[b]
            total += math.hypot(na.x - nb.x, na.y - nb.y)
        return total

    def get_escape_path(
        self, current_id: int, goal_id: int
    ) -> Optional[List[Tuple[float, float]]]:
        path_ids = self.rcg.astar(current_id, goal_id)
        if path_ids is None:
            return None
        path = []
        last = None
        min_step = max(0.05, 0.25 * self.rcg.w)
        for i in path_ids:
            if i not in self.rcg.nodes:
                continue
            pt = (self.rcg.nodes[i].x, self.rcg.nodes[i].y)
            if (
                last is not None
                and math.hypot(pt[0] - last[0], pt[1] - last[1]) < min_step
            ):
                continue
            path.append(pt)
            last = pt
        return path
