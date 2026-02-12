import math
import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .utils import (
    is_collision_free, world_to_grid, cells_in_radius, euclidean_distance,
)
from .sampling import FrontierSample
from .map_manager import MapManager


class NodeState(Enum):
    OPEN = 0      # unvisited
    CLOSED = 1    # visited by robot


@dataclass
class RCGNode:
    """A node in the Rapidly Covering Graph."""
    id: int
    x: float                     # world X
    y: float                     # world Y
    grid_row: int
    grid_col: int
    lap_index: int               # unique segment-lap id
    base_lap_index: int          # geometric lap k (before segment split)
    position_on_lap: int         # row index on the lap (for sorting)
    state: NodeState = NodeState.OPEN

    neighbors: Dict[int, float] = field(default_factory=dict)
    same_lap_prev: Optional[int] = None   # node below on same lap (lower Y)
    same_lap_next: Optional[int] = None   # node above on same lap (higher Y)
    cross_lap_neighbors: Set[int] = field(default_factory=set)


class RCG:
    def __init__(self, w: float = 0.15):
        self.w = w
        self.nodes: Dict[int, RCGNode] = {}
        self.next_id: int = 0

        self.retreat_nodes: Set[int] = set()

        self.cross_lap_max_dist = 1.5 * w

    def expand(
        self, new_samples: List[FrontierSample], map_manager: MapManager
    ) -> List[int]:
        if not new_samples:
            return []

        new_node_ids: List[int] = []
        new_node_set: Set[int] = set()

        for sample in new_samples:
            node = RCGNode(
                id=self.next_id,
                x=sample.x, y=sample.y,
                grid_row=sample.grid_row, grid_col=sample.grid_col,
                lap_index=sample.lap_index,
                base_lap_index=sample.base_lap_index,
                position_on_lap=sample.position_on_lap,
            )
            self.nodes[self.next_id] = node
            new_node_ids.append(self.next_id)
            new_node_set.add(self.next_id)
            self.next_id += 1

        self._create_same_lap_edges(map_manager, new_node_set)

        self._create_cross_lap_edges(new_node_ids, map_manager)

        return new_node_ids

    def revalidate_edges(self, map_manager: MapManager) -> int:
        if map_manager.occupancy_grid is None:
            return 0

        removed = 0
        edges_to_check: set = set()
        for nid, node in self.nodes.items():
            for nb_id in list(node.neighbors.keys()):
                edges_to_check.add((min(nid, nb_id), max(nid, nb_id)))

        for a_id, b_id in edges_to_check:
            na = self.nodes.get(a_id)
            nb = self.nodes.get(b_id)
            if na is None or nb is None:
                continue
            if not is_collision_free(
                na.grid_row, na.grid_col,
                nb.grid_row, nb.grid_col,
                map_manager.occupancy_grid,
            ):
                na.neighbors.pop(b_id, None)
                nb.neighbors.pop(a_id, None)
                na.cross_lap_neighbors.discard(b_id)
                nb.cross_lap_neighbors.discard(a_id)
                if na.same_lap_next == b_id:
                    na.same_lap_next = None
                if nb.same_lap_prev == a_id:
                    nb.same_lap_prev = None
                if nb.same_lap_next == a_id:
                    nb.same_lap_next = None
                if na.same_lap_prev == b_id:
                    na.same_lap_prev = None
                removed += 1

        return removed

    def repair_connectivity(
        self,
        map_manager: MapManager,
        anchor_id: Optional[int] = None,
        max_bridge_dist: Optional[float] = None,
    ) -> int:
        if len(self.nodes) < 2:
            return 0
        if map_manager.occupancy_grid is None:
            return 0

        if max_bridge_dist is None:
            max_bridge_dist = self.cross_lap_max_dist

        components = self._connected_components()
        if len(components) <= 1:
            return 0

        if anchor_id is None or anchor_id not in self.nodes:
            anchor_id = next(iter(self.nodes.keys()))

        anchor_comp_idx = 0
        for i, comp in enumerate(components):
            if anchor_id in comp:
                anchor_comp_idx = i
                break

        anchor_comp = set(components[anchor_comp_idx])
        others = [set(c) for i, c in enumerate(components) if i != anchor_comp_idx]

        added = 0
        progress = True
        while others and progress:
            progress = False
            for comp in list(others):
                bridge = self._find_best_bridge(
                    anchor_comp, comp, map_manager, max_bridge_dist
                )
                if bridge is None:
                    continue
                a, b, dist = bridge
                self.nodes[a].neighbors[b] = dist
                self.nodes[b].neighbors[a] = dist
                if abs(
                    self.nodes[a].base_lap_index - self.nodes[b].base_lap_index
                ) == 1:
                    self.nodes[a].cross_lap_neighbors.add(b)
                    self.nodes[b].cross_lap_neighbors.add(a)
                anchor_comp.update(comp)
                others.remove(comp)
                added += 1
                progress = True

        return added

    def _connected_components(self) -> List[Set[int]]:
        """Return connected components of the current graph."""
        unvisited: Set[int] = set(self.nodes.keys())
        comps: List[Set[int]] = []

        while unvisited:
            root = next(iter(unvisited))
            stack = [root]
            comp: Set[int] = set()
            unvisited.remove(root)

            while stack:
                nid = stack.pop()
                comp.add(nid)
                node = self.nodes.get(nid)
                if node is None:
                    continue
                for adj in node.neighbors:
                    if adj in unvisited:
                        unvisited.remove(adj)
                        stack.append(adj)

            comps.append(comp)

        return comps

    def _find_best_bridge(
        self,
        comp_a: Set[int],
        comp_b: Set[int],
        mm: MapManager,
        max_dist: float,
    ) -> Optional[tuple]:
        """Find shortest collision-free bridge edge between two components."""
        best = None
        best_d = float('inf')

        for a in comp_a:
            na = self.nodes.get(a)
            if na is None:
                continue
            for b in comp_b:
                nb = self.nodes.get(b)
                if nb is None:
                    continue
                if b in na.neighbors:
                    continue
                d = euclidean_distance(na.x, na.y, nb.x, nb.y)
                if d > max_dist or d >= best_d:
                    continue
                if abs(na.base_lap_index - nb.base_lap_index) > 1:
                    continue
                if na.base_lap_index == nb.base_lap_index:
                    continue
                if is_collision_free(
                    na.grid_row, na.grid_col,
                    nb.grid_row, nb.grid_col,
                    mm.occupancy_grid,
                ):
                    best = (a, b, d)
                    best_d = d

        return best

    def _create_same_lap_edges(
        self, map_manager: MapManager, new_ids: Set[int]
    ):
        """
        Rebuild same-lap chains for every lap that has at least one new node.
        """
        affected_laps: Set[int] = set()
        for nid in new_ids:
            affected_laps.add(self.nodes[nid].lap_index)

        laps: Dict[int, List[int]] = {}
        for nid, node in self.nodes.items():
            laps.setdefault(node.lap_index, []).append(nid)

        for lap_index in affected_laps:
            if lap_index not in laps:
                continue
            node_ids = laps[lap_index]
            node_ids.sort(key=lambda nid: self.nodes[nid].y)

            for nid in node_ids:
                self.nodes[nid].same_lap_prev = None
                self.nodes[nid].same_lap_next = None

            for i in range(len(node_ids) - 1):
                na = self.nodes[node_ids[i]]
                nb = self.nodes[node_ids[i + 1]]

                if node_ids[i] not in new_ids and node_ids[i + 1] not in new_ids:
                    na.same_lap_next = nb.id
                    nb.same_lap_prev = na.id
                    continue

                if is_collision_free(
                    na.grid_row, na.grid_col,
                    nb.grid_row, nb.grid_col,
                    map_manager.occupancy_grid,
                ):
                    cost = euclidean_distance(na.x, na.y, nb.x, nb.y)
                    na.neighbors[nb.id] = cost
                    nb.neighbors[na.id] = cost

                na.same_lap_next = nb.id
                nb.same_lap_prev = na.id

    def _create_cross_lap_edges(
        self, new_node_ids: List[int], map_manager: MapManager
    ):
        """Connect new nodes to nodes on adjacent laps (lap ± 1) within √2·w."""
        laps: Dict[int, List[int]] = {}
        for nid, node in self.nodes.items():
            laps.setdefault(node.lap_index, []).append(nid)

        for nid in new_node_ids:
            node = self.nodes[nid]
            for adj_lap in [node.base_lap_index - 1, node.base_lap_index + 1]:
                for cid in self.nodes:
                    if cid == nid:
                        continue
                    cand = self.nodes[cid]
                    if cand.base_lap_index != adj_lap:
                        continue
                    dist = euclidean_distance(node.x, node.y, cand.x, cand.y)
                    if dist > self.cross_lap_max_dist:
                        continue
                    if cid in node.neighbors:
                        continue
                    if is_collision_free(
                        node.grid_row, node.grid_col,
                        cand.grid_row, cand.grid_col,
                        map_manager.occupancy_grid,
                    ):
                        node.neighbors[cid] = dist
                        cand.neighbors[nid] = dist
                        node.cross_lap_neighbors.add(cid)
                        cand.cross_lap_neighbors.add(nid)

    def prune(
        self,
        map_manager: MapManager,
        new_node_ids: Optional[List[int]] = None,
        protected_ids: Optional[Set[int]] = None,
    ):
        if protected_ids is None:
            protected_ids = set()

        if new_node_ids is not None and len(new_node_ids) > 0:
            check_set = set(new_node_ids)
            for nid in new_node_ids:
                if nid in self.nodes:
                    check_set.update(self.nodes[nid].neighbors.keys())
        else:
            check_set = set(self.nodes.keys())

        to_remove = []
        for nid in check_set:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.state == NodeState.CLOSED:
                continue
            if nid in self.retreat_nodes:
                continue
            if nid in protected_ids:
                continue
            if not self._is_essential_node(node, map_manager):
                to_remove.append(nid)

        for nid in to_remove:
            self._remove_node(nid)

        self._prune_edges(map_manager)

    def _is_essential_node(self, node: RCGNode, mm: MapManager) -> bool:
        if node.state == NodeState.CLOSED:
            return True
        if node.id in self.retreat_nodes:
            return True

        if mm.is_adjacent_to_unknown(node.grid_row, node.grid_col, self.w):
            return True

        is_end = (node.same_lap_prev is None) or (node.same_lap_next is None)
        if is_end:
            return True

        for cx_nid in node.cross_lap_neighbors:
            cx = self.nodes.get(cx_nid)
            if cx is None:
                continue
            cx_is_end = (cx.same_lap_prev is None) or (cx.same_lap_next is None)
            if not cx_is_end:
                continue

            others = [
                n for n in cx.cross_lap_neighbors
                if n != node.id
                and n in self.nodes
                and self.nodes[n].base_lap_index == node.base_lap_index
            ]
            if len(others) == 0:
                return True

            all_non_end = all(
                (self.nodes[n].same_lap_prev is not None
                 and self.nodes[n].same_lap_next is not None)
                for n in others if n in self.nodes
            )
            if all_non_end:
                if self._is_edge_closest_to_obstacle(node, cx, others, mm):
                    return True

        return False  # inessential

    def _is_essential_edge(self, n1: RCGNode, n2: RCGNode, mm: MapManager) -> bool:
        if n1.lap_index == n2.lap_index:
            return True

        n1_end = (n1.same_lap_prev is None) or (n1.same_lap_next is None)
        n2_end = (n2.same_lap_prev is None) or (n2.same_lap_next is None)

        if n1_end and n2_end:
            return True

        if n1_end or n2_end:
            end_n = n1 if n1_end else n2
            other_n = n2 if n1_end else n1

            others = [
                n for n in end_n.cross_lap_neighbors
                if n != other_n.id
                and n in self.nodes
                and self.nodes[n].base_lap_index == other_n.base_lap_index
            ]
            if len(others) == 0:
                return True

            all_non_end = all(
                (self.nodes[n].same_lap_prev is not None
                 and self.nodes[n].same_lap_next is not None)
                for n in others if n in self.nodes
            )
            if all_non_end:
                if self._is_edge_closest_to_obstacle(other_n, end_n, others, mm):
                    return True

        return False

    def _prune_edges(self, mm: MapManager):
        edges_to_remove = []
        checked = set()
        for nid, node in self.nodes.items():
            for cx_nid in list(node.cross_lap_neighbors):
                key = (min(nid, cx_nid), max(nid, cx_nid))
                if key in checked:
                    continue
                checked.add(key)
                cx = self.nodes.get(cx_nid)
                if cx is None:
                    continue
                if not self._is_essential_edge(node, cx, mm):
                    edges_to_remove.append((nid, cx_nid))

        for a, b in edges_to_remove:
            if a in self.nodes and b in self.nodes:
                self.nodes[a].neighbors.pop(b, None)
                self.nodes[b].neighbors.pop(a, None)
                self.nodes[a].cross_lap_neighbors.discard(b)
                self.nodes[b].cross_lap_neighbors.discard(a)

    def _is_edge_closest_to_obstacle(
        self, node, cross_node, competitors, mm
    ) -> bool:
        our_dist = self._edge_obstacle_dist(node, cross_node, mm)
        for cid in competitors:
            c = self.nodes.get(cid)
            if c is None:
                continue
            if self._edge_obstacle_dist(c, cross_node, mm) < our_dist:
                return False
        return True

    def _edge_obstacle_dist(self, n1, n2, mm, samples=5) -> float:
        min_d = float('inf')
        for i in range(samples + 1):
            t = i / samples
            px = n1.x + t * (n2.x - n1.x)
            py = n1.y + t * (n2.y - n1.y)
            r, c = world_to_grid(px, py, mm.origin_x, mm.origin_y, mm.resolution)
            d = mm.distance_to_nearest_obstacle(r, c)
            min_d = min(min_d, d)
        return min_d

    def _remove_node(self, nid: int):
        if nid not in self.nodes:
            return
        node = self.nodes[nid]

        prev_id = node.same_lap_prev
        next_id = node.same_lap_next

        if prev_id is not None and prev_id in self.nodes:
            pn = self.nodes[prev_id]
            pn.same_lap_next = next_id
            pn.neighbors.pop(nid, None)
            if next_id is not None and next_id in self.nodes:
                nn = self.nodes[next_id]
                cost = euclidean_distance(pn.x, pn.y, nn.x, nn.y)
                pn.neighbors[nn.id] = cost
                nn.neighbors[pn.id] = cost

        if next_id is not None and next_id in self.nodes:
            nn = self.nodes[next_id]
            nn.same_lap_prev = prev_id
            nn.neighbors.pop(nid, None)

        for adj_id in list(node.cross_lap_neighbors):
            if adj_id in self.nodes:
                self.nodes[adj_id].neighbors.pop(nid, None)
                self.nodes[adj_id].cross_lap_neighbors.discard(nid)

        for adj_id in list(node.neighbors.keys()):
            if adj_id in self.nodes:
                self.nodes[adj_id].neighbors.pop(nid, None)

        self.retreat_nodes.discard(nid)
        del self.nodes[nid]

    def close_node(self, node_id: int):
        """Mark CLOSED.  Update retreat nodes."""
        if node_id not in self.nodes:
            return
        self.nodes[node_id].state = NodeState.CLOSED
        self.retreat_nodes.discard(node_id)

        for adj_id in self.nodes[node_id].neighbors:
            if adj_id in self.nodes and self.nodes[adj_id].state == NodeState.OPEN:
                self.retreat_nodes.add(adj_id)

    def update_retreat_nodes_near_robot(self, robot_x: float, robot_y: float):
        """Add OPEN nodes within √2·w of robot to retreat_nodes."""
        threshold = math.sqrt(2) * self.w
        for nid, node in self.nodes.items():
            if node.state == NodeState.OPEN:
                if euclidean_distance(robot_x, robot_y, node.x, node.y) <= threshold:
                    self.retreat_nodes.add(nid)
        self.retreat_nodes = {
            nid for nid in self.retreat_nodes
            if nid in self.nodes and self.nodes[nid].state == NodeState.OPEN
        }

    def auto_close_covered_nodes(self, map_manager: MapManager) -> int:
        if map_manager.covered_mask is None:
            return 0
        closed = 0
        for nid, node in list(self.nodes.items()):
            if node.state != NodeState.OPEN:
                continue
            if map_manager.is_already_covered(node.grid_row, node.grid_col):
                self.close_node(nid)
                closed += 1
        return closed

    def close_nodes_near_position(
        self,
        x: float,
        y: float,
        radius: float,
        map_manager: Optional[MapManager] = None,
    ) -> int:
        closed_count = 0
        for nid, node in list(self.nodes.items()):
            if node.state != NodeState.OPEN:
                continue
            if euclidean_distance(x, y, node.x, node.y) > radius:
                continue
            has_closed_slp = False
            for adj_id in (node.same_lap_prev, node.same_lap_next):
                if adj_id is not None and adj_id in self.nodes:
                    if self.nodes[adj_id].state == NodeState.CLOSED:
                        has_closed_slp = True
                        break
            covered_here = (
                map_manager is not None
                and map_manager.is_already_covered(node.grid_row, node.grid_col)
            )
            if has_closed_slp or covered_here:
                self.close_node(nid)
                closed_count += 1
        return closed_count

    def find_graph_path(self, from_id: int, to_id: int) -> List[int]:
        if from_id not in self.nodes or to_id not in self.nodes:
            return [to_id] if to_id in self.nodes else []
        if from_id == to_id:
            return []

        dist: Dict[int, float] = {from_id: 0.0}
        prev: Dict[int, int] = {}
        pq: list = [(0.0, from_id)]

        while pq:
            d, nid = heapq.heappop(pq)
            if nid == to_id:
                break
            if d > dist.get(nid, float('inf')):
                continue
            node = self.nodes.get(nid)
            if node is None:
                continue
            for adj_id, cost in node.neighbors.items():
                if adj_id not in self.nodes:
                    continue
                new_d = d + cost
                if new_d < dist.get(adj_id, float('inf')):
                    dist[adj_id] = new_d
                    prev[adj_id] = nid
                    heapq.heappush(pq, (new_d, adj_id))

        if to_id not in prev:
            return [to_id]  # no graph path → direct fallback

        path: List[int] = []
        nid = to_id
        while nid != from_id:
            path.append(nid)
            nid = prev.get(nid)  # type: ignore[arg-type]
            if nid is None:
                return [to_id]  # broken chain → direct fallback
        path.reverse()
        return path

    def get_nearest_retreat_node(
        self, x: float, y: float, exclude: Optional[Set[int]] = None
    ) -> Optional[RCGNode]:
        if not self.retreat_nodes:
            return None
        if exclude is None:
            exclude = set()
        nearest = None
        min_dist = float('inf')
        for nid in self.retreat_nodes:
            if nid in exclude or nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.state != NodeState.OPEN:
                continue
            d = euclidean_distance(x, y, node.x, node.y)
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest

    def get_best_retreat_node_by_path(
        self, from_id: int, exclude: Optional[Set[int]] = None
    ) -> Optional[RCGNode]:
        if from_id not in self.nodes or not self.retreat_nodes:
            return None
        if exclude is None:
            exclude = set()

        dist: Dict[int, float] = {from_id: 0.0}
        pq: list = [(0.0, from_id)]
        visited: Set[int] = set()

        while pq:
            d, nid = heapq.heappop(pq)
            if nid in visited:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                continue
            for adj_id, cost in node.neighbors.items():
                if adj_id not in self.nodes:
                    continue
                nd = d + cost
                if nd < dist.get(adj_id, float('inf')):
                    dist[adj_id] = nd
                    heapq.heappush(pq, (nd, adj_id))

        best = None
        best_d = float('inf')
        for rid in self.retreat_nodes:
            if rid in exclude or rid not in self.nodes:
                continue
            rn = self.nodes[rid]
            if rn.state != NodeState.OPEN:
                continue
            d = dist.get(rid, float('inf'))
            if d < best_d:
                best = rn
                best_d = d
        return best

    def get_nearest_open_node_by_path(
        self, from_id: int, exclude: Optional[Set[int]] = None
    ) -> Optional[RCGNode]:
        if from_id not in self.nodes:
            return None
        if exclude is None:
            exclude = set()

        dist: Dict[int, float] = {from_id: 0.0}
        pq: list = [(0.0, from_id)]
        visited: Set[int] = set()

        while pq:
            d, nid = heapq.heappop(pq)
            if nid in visited:
                continue
            visited.add(nid)
            if (nid != from_id
                    and nid not in exclude
                    and nid in self.nodes
                    and self.nodes[nid].state == NodeState.OPEN):
                return self.nodes[nid]
            node = self.nodes.get(nid)
            if node is None:
                continue
            for adj_id, cost in node.neighbors.items():
                if adj_id not in self.nodes:
                    continue
                nd = d + cost
                if nd < dist.get(adj_id, float('inf')):
                    dist[adj_id] = nd
                    heapq.heappush(pq, (nd, adj_id))

        return None   # no reachable OPEN node

    def get_nearest_node(self, x: float, y: float) -> Optional[int]:
        """Return ID of closest node to (x,y)."""
        if not self.nodes:
            return None
        nearest_id = None
        min_dist = float('inf')
        for nid, node in self.nodes.items():
            d = euclidean_distance(x, y, node.x, node.y)
            if d < min_dist:
                min_dist = d
                nearest_id = nid
        return nearest_id

    def get_nearest_open_node(self, x: float, y: float) -> Optional[RCGNode]:
        """Return closest OPEN node to (x,y)."""
        nearest = None
        min_dist = float('inf')
        for node in self.nodes.values():
            if node.state != NodeState.OPEN:
                continue
            d = euclidean_distance(x, y, node.x, node.y)
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest

    def get_open_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.state == NodeState.OPEN)

    def get_closed_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.state == NodeState.CLOSED)

    def get_node_count(self) -> int:
        return len(self.nodes)

    def get_edge_count(self) -> int:
        count = sum(len(n.neighbors) for n in self.nodes.values())
        return count // 2
