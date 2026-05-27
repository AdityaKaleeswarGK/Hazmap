"""
Progressive Sampling for the C* algorithm.

Creates sampling fronts in the newly discovered area and generates
frontier samples on parallel laps.  These samples become the nodes of
the Rapidly Covering Graph.
"""

from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import numpy as np

from .occupancy_grid_manager import OccupancyGridManager


class ProgressiveSampler:
    """
    Handles the progressive-sampling step of C*.

    At each iteration the sampler:
      1. Determines the sampling front Fi (free, unsampled area
         within the detection radius of the robot).
      2. Creates virtual laps (parallel lines) in Fi.
      3. Places frontier samples on the laps where they are adjacent
         to unknown area or obstacles.
    """

    def __init__(
        self,
        w: float,
        rd: float,
        sweep_dir: Tuple[float, float],
        ogm: OccupancyGridManager,
        known_map_mode: bool = False,
    ):
        """
        Parameters
        ----------
        w  : sampling resolution (inter-lap distance, metres)
        rd : detection / scanning radius for the sampling front
        sweep_dir : (dx, dy) unit vector perpendicular to laps
                    (laps run along sweep_dir, lap shifting is
                     perpendicular to it)
        ogm : shared OccupancyGridManager
        """
        self.w = w
        self.rd = rd
        self.ogm = ogm
        self.known_map_mode = known_map_mode
        # Sweep direction (unit vector) – laps extend along this direction
        mag = math.hypot(sweep_dir[0], sweep_dir[1]) or 1.0
        self.sweep_dx = sweep_dir[0] / mag
        self.sweep_dy = sweep_dir[1] / mag
        # Perpendicular direction (lap-shifting direction, "left to right")
        self.perp_dx = -self.sweep_dy
        self.perp_dy = self.sweep_dx

        # Tracking which cells have already been sampled
        self._sampled_cells: Set[Tuple[int, int]] = set()

        # Track last sampling position to clear cache when robot moves
        self._last_sample_x: Optional[float] = None
        self._last_sample_y: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def lap_coordinates(self, wx: float, wy: float) -> Tuple[int, float]:
        """Project a world pose onto the sampler's lap frame."""
        lap_proj = wx * self.perp_dx + wy * self.perp_dy
        sweep_proj = wx * self.sweep_dx + wy * self.sweep_dy
        lap_index = round(lap_proj / self.w)
        return lap_index, sweep_proj

    def generate_samples(
        self, robot_x: float, robot_y: float
    ) -> List[Tuple[float, float, int, float, bool]]:
        """
        Generate frontier samples around the robot's current position.

        Returns
        -------
        samples : list of (x, y, lap_index, lap_position, is_end_node)
        """
        if not self.ogm.ready:
            return []

        # Per the paper (Sec III-B2): F_i includes only the "newly
        # discovered and unsampled area".  Once a cell is sampled it must
        # stay sampled — never re-sample old positions.  The _sampled_cells
        # cache is intentionally permanent.
        self._last_sample_x = robot_x
        self._last_sample_y = robot_y

        samples: List[Tuple[float, float, int, float, bool]] = []

        # Determine range of laps to scan within detection radius
        # Project robot position onto the perp axis to get the "base lap"
        base_lap_proj = robot_x * self.perp_dx + robot_y * self.perp_dy
        n_laps = int(self.rd / self.w) + 1

        for dl in range(-n_laps, n_laps + 1):
            lap_index = round(base_lap_proj / self.w) + dl
            lap_centre_perp = lap_index * self.w

            # Position of the lap line at the robot's sweep-coordinate
            lap_ox = lap_centre_perp * self.perp_dx
            lap_oy = lap_centre_perp * self.perp_dy

            # Scan along the sweep direction within the detection radius
            sweep_proj = robot_x * self.sweep_dx + robot_y * self.sweep_dy
            n_steps = int(self.rd / self.w) + 1

            for ds in range(-n_steps, n_steps + 1):
                s_pos = sweep_proj + ds * self.w  # position along sweep axis
                sx = lap_ox + s_pos * self.sweep_dx
                sy = lap_oy + s_pos * self.sweep_dy

                # Check distance from robot
                dist = math.hypot(sx - robot_x, sy - robot_y)
                if dist > self.rd:
                    continue

                # Must be in free space
                if not self.ogm.is_free(sx, sy):
                    continue

                # Check if already sampled (discretised to nearest grid cell)
                cell_key = self._cell_key(sx, sy)
                if cell_key in self._sampled_cells:
                    continue

                # Unknown-map mode: classic frontier sampling.
                # Known-map mode: full-space coverage, so frontier adjacency is skipped.
                if not self.known_map_mode:
                    is_frontier = self.ogm.is_adjacent_to_unknown(
                        sx, sy, self.w
                    ) or self.ogm.is_adjacent_to_obstacle(sx, sy, self.w)
                    if not is_frontier:
                        continue

                # Goal Preponing: slide node backward along lap if inside the
                # raw lethal margin. Keep this just above robot_radius so we
                # do not discard samples in narrow strips along walls; Nav2's
                # planner tolerance (~0.25 m) will pull the actual goal off
                # the inflated zone when planning.
                safe_dist = 0.30  # robot_radius (0.22) + small margin
                max_shift = 0.50  # how far to slide along the lap to escape

                if self.ogm.nearest_obstacle_distance(sx, sy) < safe_dist:
                    shifted = False
                    # Pullback direction relative to robot's lap projection
                    pullback_dir = -1.0 if s_pos > sweep_proj else 1.0

                    for shift_amount in np.arange(0.05, max_shift, 0.05):
                        test_s_pos = s_pos + pullback_dir * float(shift_amount)
                        test_sx = lap_ox + test_s_pos * self.sweep_dx
                        test_sy = lap_oy + test_s_pos * self.sweep_dy

                        if (
                            self.ogm.nearest_obstacle_distance(test_sx, test_sy)
                            >= safe_dist
                        ):
                            sx, sy, s_pos = test_sx, test_sy, test_s_pos
                            shifted = True
                            break

                    if not shifted:
                        continue  # Even dragged back 0.40m it's lethal, discard this sample

                # Determine if this is an end node (at the end of a lap)
                is_end = self._is_lap_end(sx, sy)

                samples.append((sx, sy, lap_index, s_pos, is_end))
                self._sampled_cells.add(cell_key)

        return samples

    def mark_covered(self, wx: float, wy: float, radius: float) -> None:
        """
        Paint the robot's coverage swath onto the occupancy grid for stats.

        IMPORTANT: Do NOT add cells here to `_sampled_cells`. Per the paper
        (Sec III-B2), F_i is the newly discovered AND unsampled area; "sampled"
        means a frontier sample was placed at a bucket, not that the robot
        drove past it. Conflating the two locks out the entire trajectory
        swath from future frontier discovery and causes premature
        "Coverage Complete" when narrow strips along walls still need sampling.
        """
        if not self.ogm.ready:
            return

        self.ogm.mark_covered(wx, wy, radius)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cell_key(self, wx: float, wy: float) -> Tuple[int, int]:
        """Discretise a world position to a sampling-resolution bucket."""
        gx, gy = self.ogm.world_to_grid(wx, wy)
        bucket = max(1, int(self.w / self.ogm.resolution))
        return (gx // bucket, gy // bucket)

    def _prune_distant_cells(self, robot_x: float, robot_y: float) -> None:
        """
        Remove entries from _sampled_cells that are far from the
        robot's current position, allowing re-sampling when the robot
        returns to those areas later (map may have changed).
        """
        keep_radius_cells = int((self.rd * 1.5) / self.w)
        robot_key = self._cell_key(robot_x, robot_y)
        keep_radius_sq = keep_radius_cells**2

        to_remove = set()
        for cell in self._sampled_cells:
            dx = cell[0] - robot_key[0]
            dy = cell[1] - robot_key[1]
            if dx * dx + dy * dy > keep_radius_sq:
                to_remove.add(cell)

        self._sampled_cells -= to_remove

    def _is_lap_end(self, x: float, y: float) -> bool:
        """
        A sample is an end node if there is no free space further along
        the sweep direction on this lap (within w distance).
        """
        # Check forward
        fx = x + self.w * self.sweep_dx
        fy = y + self.w * self.sweep_dy
        forward_free = self.ogm.is_free(fx, fy)

        # Check backward
        bx = x - self.w * self.sweep_dx
        by = y - self.w * self.sweep_dy
        backward_free = self.ogm.is_free(bx, by)

        # End node if forward or backward is blocked
        return not forward_free or not backward_free
