"""
Rapidly Covering Graph (RCG) for the C* algorithm.

The RCG is a sparse, incrementally-built graph whose nodes = potential
waypoints and edges = potential traversal segments.  It tracks coverage
progress via Open / Closed state on each node.

Key operations (per paper):
  • expand()  – add new nodes from frontier samples and connect edges
  • prune()   – remove inessential nodes / edges to keep the graph sparse
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .occupancy_grid_manager import OccupancyGridManager


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────
class NodeState(Enum):
    OPEN = "Open"
    CLOSED = "Closed"


@dataclass
class RCGNode:
    """A node in the Rapidly Covering Graph."""

    id: int
    x: float
    y: float
    lap_index: int  # which lap this node belongs to
    lap_position: float  # position along the lap (0 = bottom)
    state: NodeState = NodeState.OPEN
    is_end_node: bool = False  # end node of its lap (terminates at obstacle/boundary)
    is_link_node: bool = False  # created by UpdateState to bridge uncovered segments
    # Local frontier-cell count at the moment this node was last CLOSED.
    # Compared against the current count by RCG.reopen_stale_closed to
    # decide whether the map has revealed enough new unknown nearby to
    # justify reopening this node for another visit.
    frontier_at_close: int = 0
    reopen_count: int = 0
    neighbors: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "left": [],  # nodes on the left adjacent lap
            "right": [],  # nodes on the right adjacent lap
            "up": [],  # node(s) above on the same lap
            "down": [],  # node(s) below on the same lap
        }
    )

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def all_neighbor_ids(self) -> Set[int]:
        s: Set[int] = set()
        for lst in self.neighbors.values():
            s.update(lst)
        return s


# ──────────────────────────────────────────────────────────────────────
# RCG Graph
# ──────────────────────────────────────────────────────────────────────
class RCG:
    """Rapidly Covering Graph."""

    def __init__(self, w: float, ogm: OccupancyGridManager):
        """
        Parameters
        ----------
        w : float   sampling resolution (distance between adjacent laps)
        ogm :       reference to the shared OccupancyGridManager
        """
        self.w = w
        self.ogm = ogm
        self.nodes: Dict[int, RCGNode] = {}
        self.edges: Set[Tuple[int, int]] = set()
        self._next_id = 0
        # Indices for fast lookup
        self._lap_nodes: Dict[
            int, List[int]
        ] = {}  # lap_index → [node_ids] sorted by lap_position
        self._node_lap_index: Dict[int, int] = {}  # node_id → lap_index for O(1) lookup
        # O(1) open/closed tracking
        self._open_ids: Set[int] = set()
        self._closed_ids: Set[int] = set()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------
    def set_node_state(self, nid: int, state: NodeState) -> None:
        """Change a node's state and update the tracking sets."""
        node = self.nodes.get(nid)
        if node is None:
            return
        if node.state == state:
            return
        # Remove from old set
        if node.state == NodeState.OPEN:
            self._open_ids.discard(nid)
        else:
            self._closed_ids.discard(nid)
        # Set new state
        node.state = state
        # Add to new set
        if state == NodeState.OPEN:
            self._open_ids.add(nid)
        else:
            self._closed_ids.add(nid)

    def add_node(
        self,
        x: float,
        y: float,
        lap_index: int,
        lap_position: float,
        is_end: bool = False,
        is_link: bool = False,
    ) -> RCGNode:
        reuse_radius = max(0.05, 0.45 * self.w)
        for nid in self._lap_nodes.get(lap_index, []):
            existing = self.nodes.get(nid)
            if existing is None:
                continue
            if math.hypot(existing.x - x, existing.y - y) <= reuse_radius:
                existing.is_end_node = existing.is_end_node or is_end
                existing.is_link_node = existing.is_link_node or is_link
                return existing

        nid = self._next_id
        self._next_id += 1
        node = RCGNode(
            id=nid,
            x=x,
            y=y,
            lap_index=lap_index,
            lap_position=lap_position,
            is_end_node=is_end,
            is_link_node=is_link,
        )
        self.nodes[nid] = node
        self._node_lap_index[nid] = lap_index
        # Track in open/closed sets
        self._open_ids.add(nid)
        self._lap_nodes.setdefault(lap_index, []).append(nid)
        # keep lap list sorted by position via bisect
        self._lap_nodes[lap_index].sort(key=lambda i: self.nodes[i].lap_position)
        return node

    def remove_node(self, nid: int) -> None:
        node = self.nodes.pop(nid, None)
        if node is None:
            return
        # Remove from tracking
        self._open_ids.discard(nid)
        self._closed_ids.discard(nid)
        self._node_lap_index.pop(nid, None)
        # Remove all edges touching this node
        to_remove = {e for e in self.edges if nid in e}
        self.edges -= to_remove
        # Remove from lap index
        if node.lap_index in self._lap_nodes:
            try:
                self._lap_nodes[node.lap_index].remove(nid)
            except ValueError:
                pass
        # Remove from neighbours of other nodes
        for nb_id in node.all_neighbor_ids():
            if nb_id in self.nodes:
                nb = self.nodes[nb_id]
                for direction, lst in nb.neighbors.items():
                    if nid in lst:
                        lst.remove(nid)

    def add_edge(self, a: int, b: int) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b))
        self.edges.add(key)

    def has_edge(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.edges

    def close_nearby_nodes(self, wx: float, wy: float, radius: float) -> List[int]:
        """
        Find all OPEN nodes within radius of (wx, wy) and mark them CLOSED.
        Returns a list of IDs for nodes that were closed in this call.
        Also snapshots the local frontier-cell count at closure so
        reopen_stale_closed can detect when the post-close map has revealed
        enough new unknown nearby to justify another visit.
        """
        closed_ids = []
        r_sq = radius**2
        snapshot_r = max(radius, self.w)
        for nid in list(self._open_ids):
            node = self.nodes.get(nid)
            if node is None:
                continue
            dist_sq = (node.x - wx) ** 2 + (node.y - wy) ** 2
            if dist_sq <= r_sq:
                node.frontier_at_close = self.ogm.find_frontier_cells_in_disk(
                    node.x, node.y, snapshot_r,
                )
                self.set_node_state(nid, NodeState.CLOSED)
                closed_ids.append(nid)
        return closed_ids

    def reopen_stale_closed(
        self,
        growth_threshold: float = 1.5,
        min_new_frontier: int = 4,
        max_reopens_per_node: int = 2,
        radius: Optional[float] = None,
    ) -> List[int]:
        """Flip CLOSED nodes back to OPEN when their local frontier has
        grown materially since closure — i.e. the OGM update revealed more
        unknown around them that wasn't visible the first time. Bounded by
        max_reopens_per_node so a single node can't ping-pong forever, and
        min_new_frontier so trivial fluctuations don't trigger.

        Returns the list of node ids that were reopened."""
        if self.ogm is None or not self.ogm.ready:
            return []
        r = max(radius if radius is not None else self.w, self.w)
        reopened: List[int] = []
        for nid in list(self._closed_ids):
            node = self.nodes.get(nid)
            if node is None:
                continue
            if node.reopen_count >= max_reopens_per_node:
                continue
            current = self.ogm.find_frontier_cells_in_disk(node.x, node.y, r)
            baseline = max(1, node.frontier_at_close)
            if (current - node.frontier_at_close) < min_new_frontier:
                continue
            if current < growth_threshold * baseline:
                continue
            node.reopen_count += 1
            node.frontier_at_close = current
            self.set_node_state(nid, NodeState.OPEN)
            reopened.append(nid)
        return reopened

    # ------------------------------------------------------------------
    # Query / Analysis
    # ------------------------------------------------------------------
    @property
    def open_nodes(self) -> List[RCGNode]:
        return [self.nodes[nid] for nid in self._open_ids if nid in self.nodes]

    @property
    def closed_nodes(self) -> List[RCGNode]:
        return [self.nodes[nid] for nid in self._closed_ids if nid in self.nodes]

    @property
    def num_open(self) -> int:
        return len(self._open_ids)

    @property
    def num_closed(self) -> int:
        return len(self._closed_ids)

    def nodes_on_lap(self, lap_index: int) -> List[RCGNode]:
        return [
            self.nodes[i] for i in self._lap_nodes.get(lap_index, []) if i in self.nodes
        ]

    def get_neighbor(self, node: RCGNode, direction: str) -> Optional[RCGNode]:
        """Get the first (nearest) neighbour in a given direction."""
        ids = node.neighbors.get(direction, [])
        for nid in ids:
            if nid in self.nodes:
                return self.nodes[nid]
        return None

    def get_open_neighbor(self, node: RCGNode, direction: str) -> Optional[RCGNode]:
        """Get the first Open neighbour in a given direction."""
        ids = node.neighbors.get(direction, [])
        for nid in ids:
            n = self.nodes.get(nid)
            if n and n.state == NodeState.OPEN:
                return n
        return None

    # ------------------------------------------------------------------
    # Expansion  (Section III-B3a of paper)
    # ------------------------------------------------------------------
    def expand(
        self, frontier_samples: List[Tuple[float, float, int, float, bool]]
    ) -> List[int]:
        """
        Expand the RCG with new nodes from frontier samples.

        Parameters
        ----------
        frontier_samples : list of (x, y, lap_index, lap_position, is_end_node)

        Returns
        -------
        new_node_ids : list of int
        """
        new_ids: List[int] = []
        for x, y, lap_idx, lap_pos, is_end in frontier_samples:
            pre_size = len(self.nodes)
            node = self.add_node(x, y, lap_idx, lap_pos, is_end=is_end)

            if len(self.nodes) == pre_size:
                continue

            new_ids.append(node.id)

        for nid in new_ids:
            self._connect_node(nid)

        return new_ids

    def _connect_node(self, nid: int) -> None:
        """Connect a node to its same-lap and cross-lap neighbours."""
        node = self.nodes[nid]
        lap_idx = self._node_lap_index.get(nid, node.lap_index)
        same_lap = self._lap_nodes.get(lap_idx, [])

        idx_in_lap = -1
        for i, eid in enumerate(same_lap):
            if eid == nid:
                idx_in_lap = i
                break

        if idx_in_lap >= 0:
            # Down neighbour
            if idx_in_lap > 0:
                down_id = same_lap[idx_in_lap - 1]
                if self._can_connect(nid, down_id):
                    self.add_edge(nid, down_id)
                    if down_id not in node.neighbors["down"]:
                        node.neighbors["down"].append(down_id)
                    if nid not in self.nodes[down_id].neighbors["up"]:
                        self.nodes[down_id].neighbors["up"].append(nid)
            # Up neighbour
            if idx_in_lap < len(same_lap) - 1:
                up_id = same_lap[idx_in_lap + 1]
                if self._can_connect(nid, up_id):
                    self.add_edge(nid, up_id)
                    if up_id not in node.neighbors["up"]:
                        node.neighbors["up"].append(up_id)
                    if nid not in self.nodes[up_id].neighbors["down"]:
                        self.nodes[up_id].neighbors["down"].append(nid)

        # Cross-lap neighbours (left / right)
        sqrt2w = math.sqrt(2) * self.w
        for adj_lap in [node.lap_index - 1, node.lap_index + 1]:
            direction = "left" if adj_lap < node.lap_index else "right"
            opp_dir = "right" if direction == "left" else "left"
            candidates = []
            for adj_nid in self._lap_nodes.get(adj_lap, []):
                adj_node = self.nodes.get(adj_nid)
                if adj_node is None:
                    continue
                dist = math.hypot(node.x - adj_node.x, node.y - adj_node.y)
                if dist <= sqrt2w and self._can_connect(nid, adj_nid):
                    lap_delta = abs(node.lap_position - adj_node.lap_position)
                    candidates.append((lap_delta, dist, adj_nid))

            if not candidates:
                continue

            candidates.sort()
            for _, _, adj_nid in candidates[:2]:
                adj_node = self.nodes[adj_nid]
                self.add_edge(nid, adj_nid)
                if adj_nid not in node.neighbors[direction]:
                    node.neighbors[direction].append(adj_nid)
                if nid not in adj_node.neighbors[opp_dir]:
                    adj_node.neighbors[opp_dir].append(nid)

    def _can_connect(self, a: int, b: int) -> bool:
        """Check collision-free path between two nodes."""
        na, nb = self.nodes[a], self.nodes[b]
        return self.ogm.is_collision_free(na.x, na.y, nb.x, nb.y)

    # ------------------------------------------------------------------
    # Pruning  (Section III-B3b of paper)
    # ------------------------------------------------------------------
    def prune(self, new_node_ids: List[int]) -> None:
        """
        Remove inessential nodes and edges.

        A node is *essential* if:
          1. It is adjacent to unknown area, OR
          2. It is an end node of its lap, OR
          3. It is in Nb(nx) where nx is an end node on an adjacent lap
             AND either (a) n is the only cross-lap neighbour of nx on
             n's lap, OR (b) edge (n, nx) is closer to obstacle/unknown.
        Everything else is inessential → prune.
        """
        to_prune: List[int] = []

        for nid in new_node_ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            if self._is_essential(node):
                continue
            to_prune.append(nid)

        for nid in to_prune:
            self._prune_node(nid)

    def _is_essential(self, node: RCGNode) -> bool:
        """Determine if a node is essential per Definition III.8."""
        # Condition 1: adjacent to unknown area
        if self.ogm.is_adjacent_to_unknown(node.x, node.y, self.w):
            return True

        # Condition 2: end node of its lap
        if node.is_end_node:
            return True

        # Condition 3: connected to an end node on an adjacent lap
        for direction in ["left", "right"]:
            for nb_id in node.neighbors.get(direction, []):
                nb = self.nodes.get(nb_id)
                if nb is None or not nb.is_end_node:
                    continue
                # nb is an end node on adjacent lap – check if node is
                # the only cross-lap neighbour of nb on node's lap
                opp_dir = "right" if direction == "left" else "left"
                nb_cross = [
                    i
                    for i in nb.neighbors.get(opp_dir, [])
                    if i in self.nodes and self.nodes[i].lap_index == node.lap_index
                ]
                if len(nb_cross) <= 1:
                    return True  # only neighbour → essential

                # Multiple neighbours – check which edge is closer to
                # obstacles / unknown
                my_dist = self._edge_obstacle_distance(node.id, nb_id)
                essential = True
                for other_id in nb_cross:
                    if other_id == node.id:
                        continue
                    other_dist = self._edge_obstacle_distance(other_id, nb_id)
                    if other_dist < my_dist:
                        essential = False
                        break
                if essential:
                    return True

        return False  # inessential

    def _edge_obstacle_distance(self, a: int, b: int) -> float:
        """
        Average distance from the edge midpoint and k intermediate
        points to the nearest obstacle.
        """
        na, nb = self.nodes[a], self.nodes[b]
        k = 5
        total = 0.0
        for i in range(k + 1):
            t = i / k
            px = na.x + t * (nb.x - na.x)
            py = na.y + t * (nb.y - na.y)
            total += self.ogm.nearest_obstacle_distance(px, py, max_range=self.w * 3)
        return total / (k + 1)

    def _prune_node(self, nid: int) -> None:
        """
        Prune an inessential node: merge its same-lap edges and
        remove its cross-lap edges.
        """
        node = self.nodes.get(nid)
        if node is None:
            return

        # Merge same-lap edges: connect up-neighbour ↔ down-neighbour
        up_ids = [i for i in node.neighbors.get("up", []) if i in self.nodes]
        down_ids = [i for i in node.neighbors.get("down", []) if i in self.nodes]
        for uid in up_ids:
            for did in down_ids:
                if self._can_connect(uid, did):
                    self.add_edge(uid, did)
                    up_n = self.nodes[uid]
                    down_n = self.nodes[did]
                    if did not in up_n.neighbors["down"]:
                        up_n.neighbors["down"].append(did)
                    if uid not in down_n.neighbors["up"]:
                        down_n.neighbors["up"].append(uid)

        self.remove_node(nid)

    # ------------------------------------------------------------------
    # A* shortest path on the RCG graph
    # ------------------------------------------------------------------
    def astar(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """A* search on the RCG. Returns list of node ids or None."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        goal = self.nodes[goal_id]

        import heapq

        open_set: list = []
        heapq.heappush(open_set, (0.0, start_id))
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start_id: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            node = self.nodes.get(current)
            if node is None:
                continue

            for nb_id in node.all_neighbor_ids():
                if nb_id not in self.nodes:
                    continue
                nb = self.nodes[nb_id]
                tentative = g_score[current] + math.hypot(node.x - nb.x, node.y - nb.y)
                if tentative < g_score.get(nb_id, float("inf")):
                    came_from[nb_id] = current
                    g_score[nb_id] = tentative
                    f = tentative + math.hypot(nb.x - goal.x, nb.y - goal.y)
                    heapq.heappush(open_set, (f, nb_id))

        return None  # no path found
