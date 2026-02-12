import numpy as np
from typing import Tuple, List, Generator


def world_to_grid(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    resolution: float
) -> Tuple[int, int]:
    col = int((world_x - origin_x) / resolution)
    row = int((world_y - origin_y) / resolution)
    return row, col


def grid_to_world(
    row: int,
    col: int,
    origin_x: float,
    origin_y: float,
    resolution: float
) -> Tuple[float, float]:
    world_x = origin_x + (col + 0.5) * resolution
    world_y = origin_y + (row + 0.5) * resolution
    return world_x, world_y


def cells_in_radius(
    center_row: int,
    center_col: int,
    radius_cells: int,
    grid_height: int,
    grid_width: int
) -> Generator[Tuple[int, int], None, None]:

    radius_sq = radius_cells * radius_cells
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_sq:
                continue
            r = center_row + dr
            c = center_col + dc
            if 0 <= r < grid_height and 0 <= c < grid_width:
                yield (r, c)


def bresenham_line(
    r0: int, c0: int,
    r1: int, c1: int
) -> List[Tuple[int, int]]:
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r, c = r0, c0
    step_r = 1 if r1 > r0 else -1
    step_c = 1 if c1 > c0 else -1

    if dc > dr:
        err = dc // 2
        while c != c1:
            cells.append((r, c))
            err -= dr
            if err < 0:
                r += step_r
                err += dc
            c += step_c
        cells.append((r, c))
    else:
        err = dr // 2
        while r != r1:
            cells.append((r, c))
            err -= dc
            if err < 0:
                c += step_c
                err += dr
            r += step_r
        cells.append((r, c))

    return cells


def is_collision_free(
    r0: int, c0: int,
    r1: int, c1: int,
    occupancy_grid: np.ndarray,
    obstacle_threshold: int = 50,
    block_unknown: bool = True,
) -> bool:
    height, width = occupancy_grid.shape
    cells = bresenham_line(r0, c0, r1, c1)

    for r, c in cells:
        if r < 0 or r >= height or c < 0 or c >= width:
            return False
        val = int(occupancy_grid[r, c])
        if val >= obstacle_threshold:
            return False
        if block_unknown and val < 0:
            return False

    return True


def find_contiguous_segments(column_bool_array: np.ndarray) -> List[Tuple[int, int]]:
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(column_bool_array):
        if val and not in_segment:
            in_segment = True
            start = i
        elif not val and in_segment:
            in_segment = False
            segments.append((start, i - 1))

    if in_segment:
        segments.append((start, len(column_bool_array) - 1))

    return segments


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
