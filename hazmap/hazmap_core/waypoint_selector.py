"""Select the next coverage waypoint using sweep-based graph traversal."""

from typing import Optional, Set, Tuple, List
from .rcg import RCG, RCGNode, NodeState
from .utils import euclidean_distance


class WaypointSelector:
    def __init__(self, w: float = 0.15):
        self.w = w
        self._sweep_forward: bool = True

    def select_next(
        self, rcg: RCG, current_node_id: int,
        skip_ids: Optional[Set[int]] = None,
    ) -> Tuple[Optional[int], bool]:
        if current_node_id not in rcg.nodes:
            return (None, True)

        _skip = skip_ids or set()
        node = rcg.nodes[current_node_id]

        fwd_id = self._walk_chain(rcg, node, forward=self._sweep_forward, skip=_skip)
        if fwd_id is not None:
            return (fwd_id, False)

        rev_id = self._walk_chain(rcg, node, forward=not self._sweep_forward, skip=_skip)
        if rev_id is not None:
            self._sweep_forward = not self._sweep_forward
            return (rev_id, False)

        cross_id = self._select_cross_lap(rcg, node, skip=_skip)
        if cross_id is not None:
            self._sweep_forward = not self._sweep_forward
            return (cross_id, False)

        return (None, True)

    def _walk_chain(
        self, rcg: RCG, start: RCGNode, forward: bool,
        skip: Optional[Set[int]] = None,
    ) -> Optional[int]:
        _skip = skip or set()
        visited: set = {start.id}
        current = start
        while True:
            ptr = current.same_lap_next if forward else current.same_lap_prev
            if ptr is None or ptr not in rcg.nodes or ptr in visited:
                return None  # end of chain or cycle guard
            visited.add(ptr)
            nxt = rcg.nodes[ptr]
            if nxt.state == NodeState.OPEN and ptr not in _skip:
                return ptr
            current = nxt  # skip CLOSED / skipped, keep walking

    def _select_cross_lap(
        self, rcg: RCG, node: RCGNode,
        skip: Optional[Set[int]] = None,
    ) -> Optional[int]:
        _skip = skip or set()
        left_cands: List[int] = []
        right_cands: List[int] = []

        for nid in node.cross_lap_neighbors:
            if nid not in rcg.nodes or nid in _skip:
                continue
            cn = rcg.nodes[nid]
            if cn.state != NodeState.OPEN:
                continue
            if cn.base_lap_index == node.base_lap_index - 1:
                left_cands.append(nid)
            elif cn.base_lap_index == node.base_lap_index + 1:
                right_cands.append(nid)

        if not left_cands and not right_cands:
            return None

        left_lap = node.base_lap_index - 1
        right_lap = node.base_lap_index + 1
        left_open = sum(
            1 for n in rcg.nodes.values()
            if n.base_lap_index == left_lap and n.state == NodeState.OPEN
        )
        right_open = sum(
            1 for n in rcg.nodes.values()
            if n.base_lap_index == right_lap and n.state == NodeState.OPEN
        )

        if left_cands and right_cands:
            chosen = left_cands if left_open <= right_open else right_cands
        elif left_cands:
            chosen = left_cands
        else:
            chosen = right_cands

        new_forward = not self._sweep_forward

        if new_forward:
            best = min(chosen, key=lambda nid: rcg.nodes[nid].y)
        else:
            best = max(chosen, key=lambda nid: rcg.nodes[nid].y)

        return best

    def update_node_state(
        self, rcg: RCG, current_id: int, next_id: Optional[int]
    ):
        if current_id not in rcg.nodes:
            return

        node = rcg.nodes[current_id]

        up_open = (
            node.same_lap_next is not None
            and node.same_lap_next in rcg.nodes
            and rcg.nodes[node.same_lap_next].state == NodeState.OPEN
        )
        down_open = (
            node.same_lap_prev is not None
            and node.same_lap_prev in rcg.nodes
            and rcg.nodes[node.same_lap_prev].state == NodeState.OPEN
        )

        if up_open and down_open:
            pass  # keep OPEN — middle of uncovered section
        else:
            rcg.close_node(current_id)

        if next_id is None or next_id not in rcg.nodes:
            return

        next_node = rcg.nodes[next_id]
        changing_lap = (next_node.lap_index != node.lap_index)

        if changing_lap and node.state == NodeState.CLOSED:
            if up_open:
                above = rcg.nodes[node.same_lap_next]
                if euclidean_distance(node.x, node.y, above.x, above.y) > self.w:
                    self._create_link_node(rcg, node, 'above')
            if down_open:
                below = rcg.nodes[node.same_lap_prev]
                if euclidean_distance(node.x, node.y, below.x, below.y) > self.w:
                    self._create_link_node(rcg, node, 'below')

    def _create_link_node(self, rcg: RCG, ref: RCGNode, direction: str,
                          resolution: float = 0.05):
        """Create a link node at distance w above/below ref to fill coverage gap."""
        row_offset = max(1, int(self.w / resolution))
        if direction == 'above':
            new_y = ref.y + self.w
            new_row = ref.grid_row + row_offset
        else:
            new_y = ref.y - self.w
            new_row = ref.grid_row - row_offset

        new_id = rcg.next_id
        rcg.next_id += 1

        link = RCGNode(
            id=new_id,
            x=ref.x, y=new_y,
            grid_row=new_row, grid_col=ref.grid_col,
            lap_index=ref.lap_index,
            base_lap_index=ref.base_lap_index,
            position_on_lap=new_row,
            state=NodeState.OPEN,
        )
        rcg.nodes[new_id] = link

        cost = self.w
        link.neighbors[ref.id] = cost
        ref.neighbors[new_id] = cost

        if direction == 'above':
            link.same_lap_prev = ref.id
            old_next = ref.same_lap_next
            if old_next is not None and old_next in rcg.nodes:
                link.same_lap_next = old_next
                rcg.nodes[old_next].same_lap_prev = new_id
                old_cost = ref.neighbors.pop(old_next, self.w)
                rcg.nodes[old_next].neighbors.pop(ref.id, None)
                new_cost = max(0.01, old_cost - self.w)
                link.neighbors[old_next] = new_cost
                rcg.nodes[old_next].neighbors[new_id] = new_cost
            ref.same_lap_next = new_id
        else:
            link.same_lap_next = ref.id
            old_prev = ref.same_lap_prev
            if old_prev is not None and old_prev in rcg.nodes:
                link.same_lap_prev = old_prev
                rcg.nodes[old_prev].same_lap_next = new_id
                old_cost = ref.neighbors.pop(old_prev, self.w)
                rcg.nodes[old_prev].neighbors.pop(ref.id, None)
                new_cost = max(0.01, old_cost - self.w)
                link.neighbors[old_prev] = new_cost
                rcg.nodes[old_prev].neighbors[new_id] = new_cost
            ref.same_lap_prev = new_id

        rcg.retreat_nodes.add(new_id)

    def reset(self):
        """Reset sweep direction to default (forward/up)."""
        self._sweep_forward = True
