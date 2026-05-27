"""
Next-Best-View (NBV) selection for information-gain coverage.

Instead of sweeping every cell in a fixed lap order, the rover repeatedly
drives to the viewpoint that *reveals* the most still-unseen area. Candidate
viewpoints are the OPEN nodes of the RCG (which already expand toward
frontiers). Utility = quality-weighted observation gain / graph travel cost.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .rcg import RCG
from .occupancy_grid_manager import OccupancyGridManager


class NextBestViewSelector:
    def __init__(
        self,
        rcg: RCG,
        ogm: OccupancyGridManager,
        sensor_range: float,
        fov_deg: float = 360.0,
        n_rays: int = 72,
        q_min: float = 0.4,
        cost_weight: float = 1.0,
        max_candidates: int = 40,
    ):
        self.rcg = rcg
        self.ogm = ogm
        self.sensor_range = sensor_range
        self.fov_deg = fov_deg
        self.n_rays = n_rays
        self.q_min = q_min
        self.cost_weight = cost_weight
        self.max_candidates = max_candidates

    def record_observation(self, wx: float, wy: float, yaw: float = 0.0) -> None:
        self.ogm.observe_from(
            wx, wy, self.sensor_range, self.fov_deg, self.n_rays, yaw=yaw
        )

    def select_view(
        self, current_id: Optional[int], robot_x: float, robot_y: float
    ) -> Tuple[Optional[int], float]:
        """Return (best_node_id, predicted_gain), or (None, 0.0) if no OPEN
        candidate would reveal anything new."""
        open_ids = list(self.rcg._open_ids)
        if not open_ids:
            return None, 0.0

        # Evaluate only the nearest max_candidates OPEN nodes (cheap bound).
        cands = []
        for nid in open_ids:
            n = self.rcg.nodes.get(nid)
            if n is None:
                continue
            d = math.hypot(n.x - robot_x, n.y - robot_y)
            cands.append((d, nid))
        cands.sort(key=lambda t: t[0])
        cands = cands[: self.max_candidates]

        best_id: Optional[int] = None
        best_utility = 0.0
        best_gain = 0.0
        for _, nid in cands:
            node = self.rcg.nodes[nid]
            gain = self.ogm.predict_observation_gain(
                node.x, node.y, self.sensor_range, self.fov_deg, self.n_rays
            )
            if gain <= 0.0:
                continue
            cost = self._travel_cost(current_id, nid, node, robot_x, robot_y)
            utility = gain / (cost ** self.cost_weight + 1e-3)
            if utility > best_utility:
                best_utility = utility
                best_id = nid
                best_gain = gain

        return best_id, best_gain

    def _travel_cost(
        self,
        current_id: Optional[int],
        nid: int,
        node,
        robot_x: float,
        robot_y: float,
    ) -> float:
        if current_id is not None and current_id in self.rcg.nodes:
            path = self.rcg.astar(current_id, nid)
            if path and len(path) >= 2:
                total = 0.0
                for a, b in zip(path, path[1:]):
                    na = self.rcg.nodes.get(a)
                    nb = self.rcg.nodes.get(b)
                    if na is not None and nb is not None:
                        total += math.hypot(na.x - nb.x, na.y - nb.y)
                return max(total, 1e-3)
        # Fallback: straight-line distance from the robot.
        return max(math.hypot(node.x - robot_x, node.y - robot_y), 1e-3)
