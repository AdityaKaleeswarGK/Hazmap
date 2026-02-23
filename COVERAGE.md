# HazMap — Coverage Path Planning: Technical Deep-Dive

> Complete technical reference for the **progressive frontier-based coverage algorithm** built on a **Reachability Connectivity Graph (RCG)** with boustrophedon sweep traversal and hybrid navigation.

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Core Concepts & Terminology](#2-core-concepts--terminology)
- [3. System Architecture](#3-system-architecture)
  - [3.1. High-Level Data Flow](#31-high-level-data-flow)
  - [3.2. Component Dependency Graph](#32-component-dependency-graph)
  - [3.3. Thread & Callback Model](#33-thread--callback-model)
- [4. Coverage Algorithm — Full Lifecycle](#4-coverage-algorithm--full-lifecycle)
  - [4.1. Initialisation Phase](#41-initialisation-phase)
  - [4.2. Outer Loop — Progressive Frontier Expansion](#42-outer-loop--progressive-frontier-expansion)
  - [4.3. Inner Loop — Boustrophedon Sweep Traversal](#43-inner-loop--boustrophedon-sweep-traversal)
  - [4.4. Dead-End Escape](#44-dead-end-escape)
  - [4.5. Termination Conditions](#45-termination-conditions)
- [5. Component Deep-Dives](#5-component-deep-dives)
  - [5.1. MapManager](#51-mapmanager)
  - [5.2. ProgressiveSampler](#52-progressivesampler)
  - [5.3. Reachability Connectivity Graph (RCG)](#53-reachability-connectivity-graph-rcg)
  - [5.4. WaypointSelector](#54-waypointselector)
  - [5.5. Navigator (Hybrid Navigation)](#55-navigator-hybrid-navigation)
- [6. Key Parameters & Their Effects](#6-key-parameters--their-effects)
- [7. Pseudocode — Essential Components](#7-pseudocode--essential-components)
  - [7.1. Main Coverage Loop](#71-main-coverage-loop)
  - [7.2. Progressive Frontier Sampling](#72-progressive-frontier-sampling)
  - [7.3. RCG Expand & Prune](#73-rcg-expand--prune)
  - [7.4. Boustrophedon Waypoint Selection](#74-boustrophedon-waypoint-selection)
  - [7.5. Hybrid Navigation](#75-hybrid-navigation)
  - [7.6. Dead-End Escape via Graph Search](#76-dead-end-escape-via-graph-search)
- [8. Coverage Masks & Frontier Dynamics](#8-coverage-masks--frontier-dynamics)
- [9. Edge Lifecycle & Graph Maintenance](#9-edge-lifecycle--graph-maintenance)
- [10. Failure Recovery Mechanisms](#10-failure-recovery-mechanisms)

---

## 1. Problem Statement

Given an **unknown, bounded 2D environment** discovered incrementally via SLAM, drive a mobile robot (TurtleBot3 Waffle) to **visit every reachable free cell** while:

- Operating on a **partial, growing occupancy grid** — the map expands as the robot explores.
- Avoiding obstacles discovered in real-time by LiDAR.
- Maintaining a **structured traversal pattern** (boustrophedon / lawn-mower) for systematic spatial coverage.
- Sampling environmental sensors along the trajectory to build hazard heatmaps.

The key challenge is that classical offline CPP (Coverage Path Planning) algorithms require the full map a priori. HazMap solves this with a **progressive, frontier-driven approach** that interleaves exploration and coverage.

---

## 2. Core Concepts & Terminology

| Term | Symbol | Definition |
|------|--------|------------|
| **Lap spacing** | `w` | Distance (metres) between adjacent parallel sweep lines. Controls coverage density. Default: `0.75m`. |
| **Sample spacing** | `δ` (delta) | Multiplier on `w` for the distance between consecutive samples on a single lap. Actual spacing = `δ × w`. |
| **Sweep direction** | — | `"x"` → vertical laps (parallel to Y-axis); `"y"` → horizontal laps (parallel to X-axis). |
| **Sampling front** | `F` | Set of occupancy grid cells that are `FREE ∧ ¬sampled` — the frontier of unexplored free space. |
| **Frontier point** | — | A free cell whose ball `B(s, w/2)` contains at least `frontier_min_cells` unknown/obstacle cells — i.e. it lies near the boundary of explored space. |
| **Lap** | `Lₖ` | A vertical (or horizontal) line at world position `x = x_start + k·w`, intersected with the sampling front. |
| **Segment** | — | A contiguous run of `TRUE` cells on a single lap column/row. Obstacles may split a lap into multiple segments. |
| **Segment-lap index** | — | A unique composite ID `= base_k × 1,000,000 + segment_idx` that distinguishes segments on the same geometric lap. |
| **RCG Node** | — | A sample point placed on the frontier, stored in the Reachability Connectivity Graph with world coordinates, grid coordinates, lap metadata, and OPEN/CLOSED state. |
| **Same-lap edge** | — | An edge connecting two adjacent nodes on the same segment-lap (sorted by position). These form the vertical sweep chains. |
| **Cross-lap edge** | — | An edge connecting two nodes on adjacent laps (`|base_k₁ - base_k₂| = 1`) within distance `1.5·w`. These enable lap transitions. |
| **OPEN node** | — | An unvisited RCG node the robot still needs to reach. |
| **CLOSED node** | — | A visited RCG node the robot has already passed through or whose area is already covered. |
| **Retreat node** | — | An OPEN node adjacent (in graph) to a CLOSED node, or within `√2·w` of the robot. Retreat nodes are potential escape targets for dead-end recovery. |
| **Coverage mask** | — | A grid-resolution boolean mask tracking which cells the robot has **physically covered** by traversing edges (marked with a brush of width `w`). |
| **Sampled mask** | — | A boolean mask tracking which cells have been **included in a sampling batch** — prevents re-sampling the same frontier region. |

---

## 3. System Architecture

### 3.1. High-Level Data Flow

```
                          ┌─────────────────────────────────┐
                          │         /map (SLAM)             │
                          │     OccupancyGrid @ ~1 Hz       │
                          └────────────┬────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          HazMapNode                                  │
│                   (hazmap_core/hazmap_node.py)                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                    MapManager                                │     │
│  │  • Stores occupancy grid, resolution, origin                 │     │
│  │  • Maintains sampled_mask (frontier consumed)                │     │
│  │  • Maintains covered_mask (physically traversed)             │     │
│  │  • Provides: is_free(), is_frontier_point(),                 │     │
│  │    is_safe_from_obstacles(), get_sampling_front_filtered()   │     │
│  │  • Re-embeds masks on SLAM map resize                        │     │
│  └──────────────┬───────────────────────────────────────────────┘     │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              ProgressiveSampler                              │     │
│  │  • Computes lap grid positions from sampling front           │     │
│  │  • Walks each lap column/row, places FrontierSamples         │     │
│  │  • Generates wall-adjacent samples near obstacle buffer band │     │
│  │  • Ensures sample-set connectivity via BFS bridging          │     │
│  │  • Returns List[FrontierSample]                              │     │
│  └──────────────┬───────────────────────────────────────────────┘     │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │           Reachability Connectivity Graph (RCG)              │     │
│  │  • expand(): create RCGNodes from samples, build edges       │     │
│  │  • prune(): remove inessential nodes/edges                   │     │
│  │  • revalidate_edges(): drop edges blocked by new obstacles   │     │
│  │  • repair_connectivity(): bridge disconnected components     │     │
│  │  • find_graph_path(): Dijkstra shortest path                 │     │
│  │  • auto_close_covered_nodes(): close nodes over covered_mask │     │
│  └──────────────┬───────────────────────────────────────────────┘     │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              WaypointSelector                                │     │
│  │  • select_next(): boustrophedon sweep along same-lap chain   │     │
│  │    1. Walk forward on same_lap_next chain                    │     │
│  │    2. If dead-end → walk reverse on same_lap_prev chain      │     │
│  │    3. If both exhausted → cross-lap transition               │     │
│  │    4. If no cross-lap → declare dead-end                     │     │
│  │  • update_node_state(): close current, create link nodes     │     │
│  └──────────────┬───────────────────────────────────────────────┘     │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              Navigator (Hybrid)                              │     │
│  │  • Direct mode: P-controller on heading + LiDAR safety       │     │
│  │    - Emergency stop at 0.22m                                 │     │
│  │    - Forward stop at 0.30m                                   │     │
│  │    - Deceleration zone at 0.50m                              │     │
│  │    - Grid-based collision check every 3 cycles               │     │
│  │  • Nav2 fallback: NavigateToPose action with timeout         │     │
│  │  • Recovery: backup + rotate toward clearer side             │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────────────┐  ┌───────────────────────────────────┐       │
│  │   SensorManager    │  │   DetectionManager (optional)     │       │
│  │  4 Hz sampling     │  │   YOLO → Depth → TF2 → pins      │       │
│  │  Gaussian splash   │  │   EMA tracking, dedup             │       │
│  └────────┬───────────┘  └──────────────┬────────────────────┘       │
│           └──────────┬──────────────────┘                            │
│                      ▼                                               │
│           ┌──────────────────────┐                                   │
│           │  Consolidated Hazard │                                   │
│           │    Impact Map        │                                   │
│           │  (priority-weighted  │                                   │
│           │   sensor + detection │                                   │
│           │   fusion)            │                                   │
│           └──────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2. Component Dependency Graph

```
                    HazMapNode
                   ╱    │    ╲
                  ╱     │     ╲
                 ╱      │      ╲
                ▼       ▼       ▼
         MapManager  Navigator  SensorManager
            │  ▲        ▲         │
            │  │        │         │
            ▼  │        │         ▼
    ProgressiveSampler  │    DetectionManager
            │           │
            ▼           │
           RCG          │
            │           │
            ▼           │
     WaypointSelector ──┘
                (feeds targets to Navigator)

    ──────── Data dependency arrows ────────
    MapManager ← /map subscription (occupancy grid)
    MapManager ← /odom subscription (robot pose)
    Navigator  ← /scan subscription (LiDAR for safety)
    Navigator  → /cmd_vel (direct motor commands)
    Navigator  → navigate_to_pose (Nav2 action client)
```

### 3.3. Thread & Callback Model

```
┌─────────────────────────────────────────────────────────────┐
│  MultiThreadedExecutor (rclpy)                              │
│                                                             │
│  Callback threads (ReentrantCallbackGroup):                 │
│    • /map subscriber        → MapManager.update_map()       │
│    • /odom subscriber       → MapManager.update_robot_pos() │
│    • /scan subscriber       → Navigator._scan_cb()          │
│    • Viz timer (2 Hz)       → _publish_viz()                │
│    • Trajectory timer (4 Hz)→ _record_trajectory()          │
│    •                           + SensorManager.sample()     │
│    • start_coverage service → spawns coverage_thread        │
│    • stop_coverage service  → sets coverage_running=False   │
│                                                             │
│  Dedicated thread:                                          │
│    • coverage_thread        → run_coverage() [blocking]     │
│      (runs the outer+inner loop, calls Navigator.go_to)     │
└─────────────────────────────────────────────────────────────┘

Key: The coverage algorithm runs in its own daemon thread to avoid
blocking the executor. All ROS callbacks (map, odom, scan, timers)
fire concurrently via the ReentrantCallbackGroup, providing the
coverage thread with continuously updated map and pose data.
```

---

## 4. Coverage Algorithm — Full Lifecycle

### 4.1. Initialisation Phase

```
 User calls /hazmap/start_coverage
            │
            ▼
 ┌─ Wait for /map (up to 30s) ────────────────────────────┐
 │  MapManager.map_received == True                        │
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
 ┌─ Wait 3s for SLAM stabilisation ───────────────────────┐
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
 ┌─ Initialise Navigator ─────────────────────────────────┐
 │  • Create Nav2 action client                           │
 │  • Subscribe to /scan for LiDAR safety                 │
 │  • Wait for Nav2 server ready (30s timeout)            │
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
 ┌─ Record start position (x₀, y₀) ──────────────────────┐
 │  ProgressiveSampler.set_start(x₀, y₀)                 │
 │  Laps are anchored: Lₖ passes through x₀ + k·w        │
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
 ┌─ 360° in-place rotation ───────────────────────────────┐
 │  Accumulates ~2π of yaw via odometry tracking          │
 │  Purpose: SLAM sees all directions → larger initial map│
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
 ┌─ Initial frontier sampling ────────────────────────────┐
 │  samples = ProgressiveSampler.progressive_sample()     │
 │  RCG.expand(samples) → creates nodes + edges           │
 │  RCG.prune() → removes inessential nodes               │
 │  current_node = RCG.get_nearest_node(x₀, y₀)          │
 └─────────────────────────────┬───────────────────────────┘
                               │
                               ▼
                    Enter Outer Loop
```

### 4.2. Outer Loop — Progressive Frontier Expansion

The outer loop is responsible for **growing the RCG** as the SLAM map expands. Each iteration:

1. **Run the inner loop** (boustrophedon sweep) until all current OPEN nodes are visited or a dead-end can't be escaped.
2. **360° rotation** for SLAM discovery of new space.
3. **Re-sample the frontier**: `ProgressiveSampler.progressive_sample()` finds new free cells beyond the current sampled mask.
4. If **new samples found**:
   - `RCG.expand(new_samples)` → adds new nodes and edges.
   - `RCG.prune()` → removes redundant nodes (not near frontiers, not chain endpoints).
   - Navigate to nearest new OPEN node.
   - Re-enter the inner loop.
5. If **no new samples** but OPEN nodes remain:
   - Attempt to reposition to the nearest reachable OPEN node via graph-path navigation.
   - Retry up to `MAX_REPOSITION_RETRIES = 3` times.
6. If **no new samples and no OPEN nodes**: **coverage complete**.

```
┌──────────────────────────────────────────────────────────┐
│                    OUTER LOOP                             │
│                                                          │
│  while coverage_running:                                 │
│      ┌────────────────────────────┐                      │
│      │     INNER LOOP             │                      │
│      │  (boustrophedon sweep)     │  ◄── see §4.3       │
│      │  until OPEN == 0 or stall  │                      │
│      └─────────────┬──────────────┘                      │
│                    │                                     │
│                    ▼                                     │
│      360° rotation for SLAM discovery                    │
│                    │                                     │
│                    ▼                                     │
│      new_samples = progressive_sample(map_manager)       │
│                    │                                     │
│            ┌───────┴────────┐                            │
│            │                │                            │
│       samples?          no samples                       │
│            │                │                            │
│            ▼                ▼                             │
│    RCG.expand()     OPEN nodes left?                     │
│    RCG.prune()         │         │                       │
│    navigate to      yes │      no │                      │
│    nearest OPEN        │         │                       │
│    continue            ▼         ▼                       │
│                  reposition    COVERAGE                   │
│                  to OPEN      COMPLETE                    │
│                  node                                    │
│                  (max 3                                   │
│                   retries)                                │
└──────────────────────────────────────────────────────────┘
```

### 4.3. Inner Loop — Boustrophedon Sweep Traversal

The inner loop drives the robot along the RCG in a lawn-mower pattern:

```
┌──────────────────────────────────────────────────────────────┐
│                    INNER LOOP (per step)                      │
│                                                              │
│  1. Auto-close nodes whose cells are already in covered_mask │
│     (map_manager.is_already_covered)                         │
│                                                              │
│  2. If OPEN == 0 → break (all nodes visited)                 │
│                                                              │
│  3. WaypointSelector.select_next(rcg, current_node):         │
│     a. Try walk forward on same_lap_next chain → OPEN node   │
│     b. Else walk reverse on same_lap_prev chain → OPEN node  │
│     c. Else cross-lap transition (prefer lap with more OPEN) │
│     d. Else → dead-end (see §4.4)                            │
│                                                              │
│  4. Navigator.go_to(target.x, target.y, prefer_direct=True)  │
│     → Direct LiDAR-safe control or Nav2 fallback             │
│                                                              │
│  5. On success:                                              │
│     • Mark edge covered: MapManager.mark_edge_covered()      │
│     • Proximity-close OPEN nodes within w/2 of robot         │
│     • Update retreat nodes near robot                        │
│     • Update current_node_id                                 │
│                                                              │
│  6. On failure:                                              │
│     • Add target to skip-set                                 │
│     • Recovery manoeuvre: backup 0.30m + rotate 0.5 rad      │
│     • If 5 consecutive failures → break inner loop (stall)   │
└──────────────────────────────────────────────────────────────┘
```

### 4.4. Dead-End Escape

When the WaypointSelector reports a dead-end (no same-lap or cross-lap OPEN neighbors):

1. **Close the current node** (it's a dead-end leaf).
2. **Find the nearest reachable OPEN node** via Dijkstra on the RCG graph: `RCG.get_nearest_open_node_by_path()`.
3. **Navigate the graph path** hop-by-hop:
   - For each node on the path, call `Navigator.go_to(node.x, node.y, prefer_direct=True)`.
   - Close each reached node, mark edges covered.
4. On arrival at the target OPEN node, resume the inner loop.

```
  Dead-end detected at node A
            │
            ▼
  Close node A
            │
            ▼
  Dijkstra: find nearest OPEN node B
  reachable from A via RCG edges
            │
       ┌────┴────┐
       │         │
    found B    no path
       │         │
       ▼         ▼
  Navigate     Break inner
  A → ... → B  loop
  (hop by hop)
       │
       ▼
  Resume sweep
  from node B
```

### 4.5. Termination Conditions

Coverage terminates when **any** of these hold:

| Condition | Trigger |
|-----------|---------|
| **Frontier exhaustion** | `progressive_sample()` returns `[]` and `get_open_count() == 0` |
| **Reposition failure** | 3 consecutive failed attempts to reach remaining OPEN nodes |
| **User stop** | `/hazmap/stop_coverage` service called → `coverage_running = False` |
| **Ctrl+C** | `KeyboardInterrupt` caught in executor → saves results and shuts down |

On termination, `SensorManager.save_results()` generates all heatmaps and the consolidated hazard impact map.

---

## 5. Component Deep-Dives

### 5.1. MapManager

**File:** `hazmap_core/map_manager.py`

Manages three co-registered grid layers on top of the SLAM occupancy grid:

```
Layer Stack (all at map resolution, typically 0.05 m/cell):

┌────────────────────────────────────────────────┐
│  occupancy_grid  (int8, from /map)             │  Values: -1 = unknown,
│                                                │          0 = free,
│                                                │          ≥50 = obstacle
├────────────────────────────────────────────────┤
│  sampled_mask    (bool, one-way OR growth)     │  True = this cell has been
│                                                │  included in a sampling batch
│                                                │  (never resets to False)
├────────────────────────────────────────────────┤
│  covered_mask    (bool, one-way OR growth)     │  True = robot physically
│                                                │  traversed an edge within
│                                                │  w/2 of this cell
└────────────────────────────────────────────────┘
```

**Key operations:**

- **`get_sampling_front_filtered()`** — Returns `(occupancy == FREE) & (~sampled_mask)` after morphological opening (3×3 kernel) and small-cluster removal. This two-layer noise filter prevents stray LiDAR rays from creating false frontiers.
- **`is_robust_frontier_point(row, col)`** — Checks that `B(cell, w/2)` contains at least `frontier_min_cells` unknown/obstacle cells. This ensures samples are placed near the genuine boundary of explored space, not in the interior.
- **`is_safe_from_obstacles(row, col)`** — Checks a circular buffer zone of radius `obstacle_buffer` around the cell. Prevents placing nodes too close to walls.
- **`mark_edge_covered(x1, y1, x2, y2)`** — Rasterises the edge using Bresenham's line algorithm, then paints a brush of width `w` (half-width on each side) into `covered_mask`. Only free cells are marked; obstacles are skipped.
- **Map resize handling** — When SLAM expands the map (origin shifts, dimensions grow), both `sampled_mask` and `covered_mask` are re-embedded into the new grid via coordinate-offset copying (`_embed_sampled_mask`, `_embed_covered_mask`).

### 5.2. ProgressiveSampler

**File:** `hazmap_core/sampling.py`

Generates `FrontierSample` objects at the boundary of explored free space. Called once per outer-loop iteration.

**Sampling pipeline:**

```
 get_sampling_front_filtered()
         │
         ▼
 _get_lap_positions()                  Determine which geometric laps Lₖ
    │                                  intersect the sampling front
    │  For each lap k:
    │    lap_x = x_start + k·w
    │    grid_col = (lap_x - origin_x) / resolution
    │    Keep only if column has TRUE cells in front
    │
    ▼
 Walk each lap column                  Place samples along each lap
    │
    │  For each contiguous segment on the lap:
    │    Assign unique segment_lap_index = k * 1M + seg_idx
    │    Walk rows with spacing = δ·w / resolution cells
    │    At each candidate position:
    │      ✓ is_free(row, col)
    │      ✓ NOT is_already_covered(row, col)
    │      ✓ is_robust_frontier_point(row, col)
    │      ✓ is_safe_from_obstacles(row, col)
    │      → Create FrontierSample
    │
    ▼
 _generate_wall_adjacent_samples()     Fill gaps near obstacle buffer band
    │
    │  • Compute distance_transform_edt from obstacle mask
    │  • Select cells in buffer band [buffer-1, buffer+2]
    │  • Filter: in sampling front, free, not near existing samples
    │  • Place with tighter mutual spacing (spacing/2)
    │
    ▼
 _ensure_connectivity()                Bridge disconnected sample clusters
    │
    │  Repeat up to 3 times:
    │    • Build adjacency graph (same-lap sequential + cross-lap within 1.5·w)
    │    • Find connected components
    │    • If >1 component: BFS free-space path between closest pair
    │    • Place bridge samples along the BFS path
    │
    ▼
 _build_coverage_mask()                Mark the covered area in sampled_mask
    │
    │  For each sample: stamp a w/2-radius square in the mask
    │  sampled_mask |= (sampling_front & stamp_mask)
    │
    ▼
 Return List[FrontierSample]
```

**Wall-adjacent samples** solve a key problem: laps spaced at `w` can leave narrow corridors along walls uncovered. The EDT (Euclidean Distance Transform) identifies cells exactly at the obstacle buffer distance, and extra samples are placed there.

### 5.3. Reachability Connectivity Graph (RCG)

**File:** `hazmap_core/rcg.py`

The RCG is the central data structure — a **sparse, dynamically growing graph** that organises frontier samples into a navigable network.

**Node structure:**

```
RCGNode:
  id:                int           Unique auto-increment ID
  x, y:             float          World coordinates (metres)
  grid_row, grid_col: int          Occupancy grid coordinates
  lap_index:         int           Segment-lap composite ID
  base_lap_index:    int           Geometric lap k
  position_on_lap:   int           Position for sorting along the lap
  state:             OPEN|CLOSED   Visit status
  neighbors:         Dict[id→cost] Adjacency list with Euclidean cost
  same_lap_prev:     Optional[id]  Previous node on the same-lap chain (lower Y)
  same_lap_next:     Optional[id]  Next node on the same-lap chain (higher Y)
  cross_lap_neighbors: Set[id]     Nodes on adjacent laps (base_k ± 1)
```

**Edge types:**

```
                Lap k-1        Lap k          Lap k+1
                  │              │               │
                  │    cross     │    cross      │
                  │◄────────────►│◄─────────────►│
                  │              │               │
               ●──┼──────────●──┼────────●──────┼──●
               │  │          │  │        │      │  │
    same-lap   │  │          │  │        │      │  │  same-lap
    edges      │  │          │  │        │      │  │  edges
    (vertical) │  │          │  │        │      │  │  (vertical)
               │  │          │  │        │      │  │
               ●──┼──────────●──┼────────●──────┼──●
               │  │             │               │
               ●  │             │               │
                  │             │               │
    Same-lap edges: sequential along one segment (collision-free verified)
    Cross-lap edges: between nodes on laps k and k±1, dist ≤ 1.5·w
```

**RCG operations:**

| Operation | Description |
|-----------|-------------|
| `expand(samples)` | Creates RCGNodes, builds same-lap chains (sorted by Y, collision-free edges), creates cross-lap edges (base_k ± 1, dist ≤ 1.5·w, collision-free) |
| `prune(new_ids, protected)` | Removes OPEN nodes that are not essential: not near unknown space, not chain endpoints, not sole cross-lap bridges. Re-links chain pointers. Also prunes non-essential cross-lap edges. |
| `revalidate_edges()` | Re-checks all edges against the current occupancy grid. Removes edges that now cross obstacles (map updated by SLAM). |
| `repair_connectivity(anchor_id)` | Finds connected components. If >1, bridges them by finding the shortest collision-free cross-lap edge between components. |
| `auto_close_covered_nodes()` | Iterates OPEN nodes; if `covered_mask[row, col]` is True, closes them. |
| `close_nodes_near_position(x, y, r)` | Closes OPEN nodes within radius `r` that have a CLOSED same-lap neighbor or are in `covered_mask`. |
| `find_graph_path(from, to)` | Dijkstra shortest path on the weighted RCG. Returns list of node IDs. |

### 5.4. WaypointSelector

**File:** `hazmap_core/waypoint_selector.py`

Implements the **boustrophedon sweep** pattern on the RCG:

```
                  Lap k                       Lap k+1
                    │                           │
                    │  ● ← start                │
                    │  │                        │
              ┌─────│──▼── sweep forward ──┐    │
              │     │  ●                   │    │
              │     │  │                   │    │
              │     │  ●                   │    │
              │     │  │                   │    │
              │     │  ● ← end of chain    │    │
              │     │                      │    │
              │     │      cross-lap       │    │
              │     │   ─ ─ ─ ─ ─ ─ ─ ─ ─►│──● │
              │     │                      │  │ │
              │     │                      │  ▼ │
              │     │      sweep reverse   │  ● │
              │     │                      │  │ │
              │     │                      │  ▼ │
              │     │                      │  ● │
              └─────│──────────────────────│────┘
                    │                      │
```

**Selection algorithm (priority order):**

1. **Walk forward** on `same_lap_next` chain — skip CLOSED, find first OPEN.
2. **Walk reverse** on `same_lap_prev` chain — flip sweep direction, find first OPEN.
3. **Cross-lap transition** — pick adjacent lap with more OPEN nodes. If `new_forward` is True, pick the node with lowest Y (start from bottom); else highest Y (start from top).
4. **Dead-end** — no OPEN neighbor reachable. Return `(None, is_dead_end=True)`.

**Link node creation:** When transitioning cross-lap, if the current node is being closed but has uncovered same-lap neighbors more than `w` away, a **link node** is injected at distance `w` to fill the coverage gap. This ensures the area between the departure point and the next uncovered node gets visited.

### 5.5. Navigator (Hybrid Navigation)

**File:** `hazmap_core/navigator.py`

Two-tier navigation system:

```
┌─────────────────────────────────────────────────────────┐
│                    go_to(x, y)                          │
│                                                         │
│     prefer_direct=True?                                 │
│         │           │                                   │
│        yes          no                                  │
│         │           │                                   │
│         ▼           │                                   │
│   _go_to_direct()   │                                   │
│         │           │                                   │
│    ┌────┴────┐      │                                   │
│  success   fail     │                                   │
│    │         │      │                                   │
│    │    fallback?   │                                   │
│    │    ┌────┴──┐   │                                   │
│    │   yes     no   │                                   │
│    │    │      │    │                                   │
│    │    ▼      ▼    ▼                                   │
│    │  _go_to_nav2() return False                        │
│    │    │                                               │
│    ▼    ▼                                               │
│  return result                                          │
└─────────────────────────────────────────────────────────┘
```

**Direct navigation controller:**

- **P-controller** on heading error: `ω = clamp(k_ang × heading_err, ±angular_speed)`
- **Speed modulation**: `v = linear_speed × cos(heading_err) × min(1, dist/decel_zone)`
- **LiDAR safety** (front ±35° arc):
  - `< 0.22m` → **emergency stop**, abort immediately
  - `< 0.30m` → **forward stop**, allow rotation only; abort after 1.5s blocked
  - `< 0.50m` → **deceleration zone**, scale speed linearly
- **Grid safety check** every 3 cycles: verify Bresenham line from robot to target is collision-free on occupancy grid.
- **Arrival**: `distance < xy_tolerance (0.08m)`.

---

## 6. Key Parameters & Their Effects

| Parameter | Default | Effect on Coverage |
|-----------|---------|-------------------|
| `w` | 0.75m | **Lap spacing.** Smaller → denser coverage, more laps, longer runtime. Larger → faster but may miss narrow features. Should be ≤ sensor detection radius. |
| `delta` | 1 | **Sample spacing multiplier.** At `δ=1`, samples are placed every `w` metres along each lap. Higher values skip cells, reducing graph density. |
| `sweep_direction` | `"x"` | `"x"` creates vertical laps (columns); `"y"` creates horizontal laps (rows). Choose based on expected environment geometry. |
| `obstacle_buffer` | 0.375m | **Minimum clearance from obstacles** for sample placement. Prevents nodes from being placed in tight gaps the robot can't safely navigate. |
| `frontier_min_cells` | 2 | **Robustness filter.** A point is only a frontier if its ball contains ≥ this many unknown/obstacle cells. Higher = more conservative (ignores thin frontier strips). |
| `direct_nav_max_distance` | 0.70m | Edges longer than this always use Nav2. Shorter → more Nav2 calls (slower but safer). |
| `direct_nav_min_clearance` | 0.18m | If either endpoint of an edge is closer than this to an obstacle, direct nav is skipped. |
| `direct_nav_linear_speed` | 0.20 m/s | Robot forward speed during direct control. |
| `direct_nav_fallback_to_nav2` | true | If false, a failed direct navigation is not retried with Nav2 — the step is simply skipped. |

---

## 7. Pseudocode — Essential Components

### 7.1. Main Coverage Loop

```
function RUN_COVERAGE():
    wait_for_map()
    wait(3s)  // SLAM stabilisation
    
    navigator = Navigator(...)
    sampler.set_start(robot.x, robot.y)
    navigator.rotate_360()
    
    samples = sampler.progressive_sample(map_manager)
    rcg.expand(samples, map_manager)
    rcg.prune(map_manager, new_ids)
    current = rcg.get_nearest_node(robot.x, robot.y)
    
    reposition_retries = 0
    
    while coverage_running:
        // ══════ INNER LOOP ══════
        while coverage_running:
            rcg.auto_close_covered_nodes(map_manager)
            if rcg.open_count == 0:
                break
            
            next_id, is_dead_end = selector.select_next(rcg, current)
            
            if is_dead_end:
                rcg.close_node(current)
                target = rcg.nearest_open_by_path(current)
                if target is None:
                    break  // inner exhausted
                navigate_graph_path(current, target.id)
                continue
            
            selector.update_node_state(rcg, current, next_id)
            success = navigator.go_to(target.x, target.y, prefer_direct=True)
            
            if success:
                map_manager.mark_edge_covered(prev.x, prev.y, target.x, target.y)
                rcg.close_nodes_near_position(robot.x, robot.y, w/2)
                rcg.update_retreat_nodes(robot.x, robot.y)
                current = next_id
            else:
                skip_set.add(next_id)
                navigator.backup(0.30m, rotate=0.5rad)
                if consecutive_fails >= 5:
                    break  // stalled
        
        // ══════ FRONTIER EXPANSION ══════
        navigator.rotate_360()
        new_samples = sampler.progressive_sample(map_manager)
        
        if not new_samples:
            if rcg.open_count > 0:
                // try reposition (max 3 retries)
                target = rcg.nearest_open_by_path(current)
                navigate_graph_path(current, target.id)
                reposition_retries += 1
                if reposition_retries > 3:
                    break  // give up
                continue
            else:
                break  // COVERAGE COMPLETE
        
        rcg.expand(new_samples, map_manager)
        rcg.prune(map_manager, new_ids, protected={current})
        current = rcg.get_nearest_open_node(robot.x, robot.y)
    
    sensor_manager.save_results()
```

### 7.2. Progressive Frontier Sampling

```
function PROGRESSIVE_SAMPLE(map_manager) → List[FrontierSample]:
    front = map_manager.get_sampling_front_filtered()
    //  front = (occupancy == FREE) & (~sampled_mask)
    //  after morphological opening + small-cluster removal
    
    if front is empty:
        return []
    
    laps = GET_LAP_POSITIONS(front, map_manager)
    //  For sweep_direction "x":
    //    For each k where lap_x = start_x + k·w intersects front:
    //      laps.append((k, grid_column))
    
    spacing = max(1, floor(δ·w / resolution))
    all_samples = []
    
    for (k, grid_col) in laps:
        column = front[:, grid_col]
        segments = find_contiguous_segments(column)
        
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_lap_id = k * 1_000_000 + seg_idx
            pos = seg_start
            last_placed = seg_start - spacing
            
            while pos <= seg_end:
                row, col = pos, grid_col
                
                if NOT map_manager.is_free(row, col):              skip
                if map_manager.is_already_covered(row, col):       skip
                if (pos - last_placed) < spacing:                  skip
                if NOT is_robust_frontier_point(row, col):         skip
                if NOT is_safe_from_obstacles(row, col):           skip
                
                world_x, world_y = grid_to_world(row, col)
                all_samples.append(FrontierSample(
                    x=world_x, y=world_y,
                    grid_row=row, grid_col=col,
                    lap_index=seg_lap_id,
                    base_lap_index=k,
                    position_on_lap=pos
                ))
                last_placed = pos
                pos += spacing
    
    wall_samples = GENERATE_WALL_ADJACENT_SAMPLES(front, all_samples, spacing)
    all_samples.extend(wall_samples)
    
    all_samples = ENSURE_CONNECTIVITY(all_samples, map_manager, spacing)
    //  Build adjacency graph → find components → BFS bridge → repeat 3x
    
    coverage_mask = BUILD_COVERAGE_MASK(all_samples, front)
    map_manager.sampled_mask |= coverage_mask
    
    return all_samples
```

### 7.3. RCG Expand & Prune

```
function RCG_EXPAND(samples, map_manager) → List[int]:
    new_ids = []
    for sample in samples:
        node = RCGNode(id=next_id++, sample.x, sample.y, ...)
        nodes[node.id] = node
        new_ids.append(node.id)
    
    // ── Same-lap edges ──
    for each affected lap_index:
        lap_nodes = [n for n in nodes if n.lap_index == lap_index]
        sort lap_nodes by y-coordinate
        
        for i in 0..len(lap_nodes)-2:
            na, nb = lap_nodes[i], lap_nodes[i+1]
            na.same_lap_next = nb.id
            nb.same_lap_prev = na.id
            if is_collision_free(na.grid, nb.grid, occupancy_grid):
                cost = euclidean_distance(na, nb)
                na.neighbors[nb.id] = cost
                nb.neighbors[na.id] = cost
    
    // ── Cross-lap edges ──
    for each new node n:
        for each existing node c where |c.base_lap - n.base_lap| == 1:
            dist = euclidean_distance(n, c)
            if dist ≤ 1.5·w AND c.id not in n.neighbors:
                if is_collision_free(n.grid, c.grid, occupancy_grid):
                    n.neighbors[c.id] = dist
                    c.neighbors[n.id] = dist
                    n.cross_lap_neighbors.add(c.id)
                    c.cross_lap_neighbors.add(n.id)
    
    return new_ids

function RCG_PRUNE(map_manager, new_ids, protected):
    check_set = new_ids ∪ {all neighbors of new_ids}
    
    to_remove = []
    for nid in check_set:
        node = nodes[nid]
        if node.state == CLOSED: continue
        if nid in retreat_nodes: continue
        if nid in protected: continue
        
        if NOT IS_ESSENTIAL(node, map_manager):
            to_remove.append(nid)
    
    for nid in to_remove:
        REMOVE_NODE(nid)  // re-link chain pointers, remove from neighbors
    
    PRUNE_EDGES(map_manager)  // remove non-essential cross-lap edges

function IS_ESSENTIAL(node, mm) → bool:
    if node is CLOSED or in retreat_nodes: return True
    if mm.is_adjacent_to_unknown(node, w):  return True   // near frontier
    if node is chain endpoint (prev==None or next==None): return True
    if node is sole cross-lap bridge to an endpoint:       return True
    return False
```

### 7.4. Boustrophedon Waypoint Selection

```
function SELECT_NEXT(rcg, current_id) → (next_id, is_dead_end):
    node = rcg.nodes[current_id]
    
    // 1. Walk forward along same-lap chain
    fwd = WALK_CHAIN(rcg, node, forward=sweep_forward)
    if fwd is not None:
        return (fwd, false)
    
    // 2. Walk reverse
    rev = WALK_CHAIN(rcg, node, forward=NOT sweep_forward)
    if rev is not None:
        sweep_forward = NOT sweep_forward   // flip direction
        return (rev, false)
    
    // 3. Cross-lap transition
    cross = SELECT_CROSS_LAP(rcg, node)
    if cross is not None:
        sweep_forward = NOT sweep_forward   // flip for new lap
        return (cross, false)
    
    // 4. Dead-end
    return (None, true)

function WALK_CHAIN(rcg, start, forward) → Optional[int]:
    current = start
    visited = {start.id}
    while True:
        ptr = current.same_lap_next if forward else current.same_lap_prev
        if ptr is None or ptr in visited:
            return None  // end of chain
        visited.add(ptr)
        if rcg.nodes[ptr].state == OPEN and ptr not in skip_set:
            return ptr
        current = rcg.nodes[ptr]  // skip CLOSED, keep walking

function SELECT_CROSS_LAP(rcg, node) → Optional[int]:
    left_cands  = [n for n in node.cross_lap_neighbors
                   if n.base_lap == node.base_lap - 1 and n.state == OPEN]
    right_cands = [n for n in node.cross_lap_neighbors
                   if n.base_lap == node.base_lap + 1 and n.state == OPEN]
    
    if neither:
        return None
    
    // Prefer the lap with fewer remaining OPEN nodes (complete it first)
    left_open  = count(OPEN nodes on lap base_lap - 1)
    right_open = count(OPEN nodes on lap base_lap + 1)
    chosen = left_cands if left_open ≤ right_open else right_cands
    
    // Pick entry point based on new sweep direction
    new_forward = NOT sweep_forward
    if new_forward:
        return min(chosen, key=y)   // start from bottom
    else:
        return max(chosen, key=y)   // start from top
```

### 7.5. Hybrid Navigation

```
function GO_TO_DIRECT(target_x, target_y, timeout) → bool:
    t0 = now()
    
    while (now() - t0) < timeout:
        rx, ry, ryaw = get_robot_pose()
        dx, dy = target_x - rx, target_y - ry
        dist = hypot(dx, dy)
        
        if dist ≤ xy_tolerance:
            stop_robot()
            return True   // ARRIVED
        
        // ── Heading control ──
        target_heading = atan2(dy, dx)
        heading_err = normalize(target_heading - ryaw)
        ω = clamp(2.0 × heading_err, ±angular_speed)
        
        // ── Speed control ──
        heading_factor = max(0, cos(heading_err))
        dist_factor = min(1, dist / 0.25)
        v = linear_speed × heading_factor × dist_factor
        if v < 0.015: v = 0   // pure rotation deadband
        
        // ── LiDAR safety (front ±35°) ──
        front_min = min(scan.ranges in front arc)
        
        if front_min < 0.22m:
            stop_robot()
            return False   // EMERGENCY
        
        if front_min < 0.30m:
            v = 0   // stop forward, rotate only
            if blocked > 1.5s:
                stop_robot()
                return False   // BLOCKED
        
        elif front_min < 0.50m:
            slow = (front_min - 0.30) / (0.50 - 0.30)
            v *= clamp(slow, 0.2, 1.0)
        
        // ── Grid safety (every 3 cycles) ──
        if NOT is_collision_free(robot_grid, target_grid, occupancy_grid):
            stop_robot()
            return False
        
        publish(v, ω)
        sleep(50ms)
    
    stop_robot()
    return False   // TIMEOUT
```

### 7.6. Dead-End Escape via Graph Search

```
function NAVIGATE_GRAPH_PATH(from_id, to_id):
    path = RCG.find_graph_path(from_id, to_id)
    //  Dijkstra on the weighted RCG adjacency list
    //  Returns: list of node IDs [hop1, hop2, ..., to_id]
    //  (excludes from_id)
    
    for each node_id in path:
        if NOT coverage_running: break
        node = rcg.nodes[node_id]
        
        success = navigator.go_to(node.x, node.y, prefer_direct=True)
        
        if success:
            mark_edge_covered(current → node)
            if node.state == OPEN:
                rcg.close_node(node_id)
            current = node_id
            rcg.close_nodes_near_position(robot.x, robot.y, w/2)
            rcg.update_retreat_nodes(robot.x, robot.y)
        else:
            skip_set.add(node_id)
            // skip this hop, try next

function DIJKSTRA_GRAPH_PATH(from_id, to_id) → List[int]:
    dist = {from_id: 0}
    prev = {}
    pq = [(0, from_id)]
    
    while pq not empty:
        d, nid = heappop(pq)
        if nid == to_id: break
        if d > dist[nid]: continue
        
        for adj_id, cost in nodes[nid].neighbors:
            new_d = d + cost
            if new_d < dist.get(adj_id, ∞):
                dist[adj_id] = new_d
                prev[adj_id] = nid
                heappush(pq, (new_d, adj_id))
    
    // Reconstruct path
    path = []
    nid = to_id
    while nid ≠ from_id:
        path.prepend(nid)
        nid = prev[nid]
    return path
```

---

## 8. Coverage Masks & Frontier Dynamics

The interaction between the two boolean masks is critical to understanding how coverage progresses:

```
    Time ──────────────────────────────────────────────►

    ┌─────────────────────────────────────────────────────┐
    │                   SLAM Map Growth                    │
    │                                                     │
    │  t=0:    ████░░░░░░░░░░     (█=unknown, ░=free)     │
    │  t=10:   ██████░░░░░░░░░░░░                         │
    │  t=20:   ████████░░░░░░░░░░░░░░░░                   │
    └─────────────────────────────────────────────────────┘
    
    Sampling Front = FREE ∧ ¬sampled_mask
    ┌─────────────────────────────────────────────────────┐
    │  When sampler runs at t=5:                           │
    │    sampling_front = all current free cells            │
    │    → places samples → builds coverage_mask            │
    │    → sampled_mask |= coverage_mask                    │
    │                                                     │
    │  At t=15 (map grew):                                 │
    │    New free cells appeared beyond old sampled_mask    │
    │    sampling_front = new_free_cells only               │
    │    → places samples in the NEW region only            │
    │    This is the "progressive" aspect.                  │
    └─────────────────────────────────────────────────────┘
    
    Covered Mask = physically traversed
    ┌─────────────────────────────────────────────────────┐
    │  As robot moves along edge (A → B):                  │
    │    Bresenham rasterise the edge                       │
    │    Paint brush of width w along the line              │
    │    covered_mask[cells] = True                         │
    │                                                     │
    │  auto_close_covered_nodes() uses this:               │
    │    if covered_mask[node.row, node.col]:               │
    │      close the node (already physically covered)     │
    └─────────────────────────────────────────────────────┘
```

---

## 9. Edge Lifecycle & Graph Maintenance

RCG edges are not static — the SLAM map evolves, and newly discovered obstacles may invalidate previously valid edges.

```
    Edge Creation           Edge Validation          Edge Removal
    ─────────────           ───────────────          ────────────
    
    expand():               revalidate_edges():      prune():
    Bresenham collision     Re-check ALL edges       Remove non-essential
    check at creation       against current          cross-lap edges
    time                    occupancy grid           (edge pruning)
         │                       │                        │
         ▼                       ▼                        ▼
    Edge added to           If obstacle now           Edge removed from
    both nodes'             blocks the edge:          neighbors + cross_lap
    neighbor dicts          remove from both          + chain pointers
                            endpoints
                                 │
                                 ▼
                            repair_connectivity():
                            If removal disconnected
                            the graph, find shortest
                            collision-free bridge
                            between components
    
    ──── Timing ────
    revalidate_edges() runs on the viz timer (2 Hz)
    repair_connectivity() runs immediately after revalidation
    prune() runs once per outer-loop iteration after expand()
```

---

## 10. Failure Recovery Mechanisms

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Direct nav obstacle** | LiDAR front arc < 0.22m | Emergency stop → return fail → Nav2 fallback |
| **Direct nav blocked** | Forward stopped > 1.5s | Abort → Nav2 fallback |
| **Direct nav timeout** | > `direct_timeout` seconds | Abort → Nav2 fallback |
| **Grid safety fail** | Bresenham check fails on occupancy grid | Abort → Nav2 fallback |
| **Nav2 rejected** | Goal rejected by Nav2 planner | Mark target as skipped |
| **Nav2 timeout** | > 120s without reaching goal | Cancel goal, mark as skipped |
| **Consecutive failures** | 5 consecutive go_to failures in inner loop | Break inner loop → outer loop re-samples |
| **Dead-end** | No OPEN same-lap or cross-lap neighbor | Dijkstra escape to nearest reachable OPEN node |
| **No frontier** | `progressive_sample()` returns `[]` | 360° rotation → re-sample. If still none + OPEN nodes remain → reposition (max 3 retries) |
| **Stale edges** | SLAM reveals new obstacle on existing edge | `revalidate_edges()` removes edge → `repair_connectivity()` bridges if needed |
| **Map resize** | SLAM origin shifts or dimensions grow | `sampled_mask` and `covered_mask` re-embedded into new grid coordinates |

```
    Navigation Failure Cascade:
    
    go_to(target, prefer_direct=True)
        │
        ▼
    _go_to_direct()
        │
    ┌───┴───┐
    OK     FAIL
    │       │
    │       ▼
    │   fallback_to_nav2?
    │   ┌───┴───┐
    │  yes      no
    │   │       │
    │   ▼       ▼
    │  _go_to_nav2()  return False
    │   │              │
    │ ┌─┴─┐            │
    │ OK  FAIL         │
    │ │    │           │
    │ │    ▼           │
    │ │  consecutive_fails++
    │ │  skip_set.add(target)
    │ │  backup(0.30m) + rotate(0.5 rad)
    │ │    │
    │ │    ▼
    │ │  if fails ≥ 5:
    │ │    break inner loop
    │ │    → outer loop re-samples
    ▼ ▼
    continue sweep
```

---

