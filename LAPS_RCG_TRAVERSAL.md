# Virtual Laps, RCG Construction & Graph Traversal

> Focused technical reference for the three core mechanics: **how laps are laid out**, **how the RCG is built from them**, and **how the robot decides where to move** (including Dijkstra-based escape).

---

## 1. Virtual Laps

### What They Are

Laps are **imaginary parallel lines** spaced `w` metres apart, anchored at the robot's start position. They define where the robot should sweep.

- `sweep_direction = "x"` → laps are **vertical** lines (constant X, varying Y).
- `sweep_direction = "y"` → laps are **horizontal** lines (constant Y, varying X).

Lap `k` sits at world position:

```
lap_x = start_x + k * w       (for sweep_direction "x")
lap_y = start_y + k * w       (for sweep_direction "y")
```

`k` ranges over all integers where the lap intersects the current **sampling front** (free cells not yet sampled).

### How Lap Positions Are Computed

```
sampling_front = (occupancy_grid == FREE) & (~sampled_mask)
                 → morphological opening (3×3)
                 → remove clusters < min_cluster cells

For sweep "x":
    active_cols = columns where sampling_front has any TRUE cell
    min_x, max_x = world X range of active columns
    k_min = floor((min_x - start_x) / w)
    k_max = ceil((max_x - start_x) / w)

    for k in [k_min .. k_max]:
        grid_col = (start_x + k*w - origin_x) / resolution
        if sampling_front[:, grid_col] has any TRUE → emit (k, grid_col)
```

### Segments Within a Lap

A single lap column may have **gaps** — obstacles splitting the free space. Each contiguous run of TRUE cells is a **segment**.

```
Lap k column (top to bottom):
    ░░░░░░░░░░░     ← unknown / obstacle
    ████████████     ← segment 1: rows 20–45
    ░░░░░░░░░░░     ← obstacle gap
    ████████████     ← segment 2: rows 60–90
    ░░░░░░░░░░░     ← unknown

Each segment gets a unique segment-lap ID:
    seg_lap_id = k * 1_000_000 + segment_index
```

This ensures nodes on the same geometric lap but separated by an obstacle are **never** chained together as same-lap neighbors.

### Sample Placement On a Lap

Walk each segment from start to end. Place a `FrontierSample` every `δ·w / resolution` cells, subject to:

1. Cell is `FREE` on the occupancy grid.
2. Cell is **not** already in `covered_mask` (physically traversed).
3. Cell is a **robust frontier point** — `B(cell, w/2)` contains ≥ `frontier_min_cells` unknown/obstacle cells.
4. Cell is **safe from obstacles** — no obstacle within `obstacle_buffer` radius.

```
             Lap k-1          Lap k          Lap k+1
               │                │               │
               │     w          │      w        │
               │◄──────────────►│◄─────────────►│
               │                │               │
               │                ● sample 0      │
               │                │               │
               │                ● sample 1      │
               │                │  ← spacing = δ·w
               │                ● sample 2      │
               │                │               │
               │            ████████ obstacle   │
               │                │               │
               │                ● sample 3      │  ← new segment
               │                │               │
               │                ● sample 4      │
```

### Wall-Adjacent Samples

Regular lap samples can miss narrow corridors along walls. Extra samples are placed in the **obstacle buffer band**:

```
Euclidean Distance Transform from obstacle mask:

    ████████████████   obstacle
    ░ 1 2 3 4 5 ...   distance in cells
          ▲
          │
    buffer band = cells at distance [buffer-1, buffer+2]

Candidates = buffer_band ∩ sampling_front ∩ FREE
           − exclusion zone around existing samples
```

These are assigned to the nearest existing segment-lap and treated identically in the RCG.

---

## 2. RCG Construction

### Node Data Structure

```
RCGNode:
    id:               int          auto-increment
    x, y:             float        world metres
    grid_row, col:    int          occupancy grid cell
    lap_index:        int          segment-lap composite ID
    base_lap_index:   int          geometric lap k
    position_on_lap:  int          row position (for sorting)
    state:            OPEN|CLOSED
    neighbors:        {node_id → euclidean_cost}
    same_lap_prev:    Optional[id]   ← lower Y on same segment
    same_lap_next:    Optional[id]   ← higher Y on same segment
    cross_lap_neighbors: Set[id]     ← nodes on adjacent laps (k±1)
```

### expand() — Building the Graph

Called each time new frontier samples are produced:

**Step 1: Create nodes**

```
for each FrontierSample:
    create RCGNode with next_id++
    store in nodes dict
```

**Step 2: Same-lap edges (vertical chains)**

For every segment-lap that contains at least one new node:

```
lap_nodes = all nodes with this lap_index
sort by y coordinate (ascending)

for consecutive pairs (A, B):
    A.same_lap_next = B.id
    B.same_lap_prev = A.id
    if Bresenham line A→B is collision-free on occupancy grid:
        cost = euclidean_distance(A, B)
        A.neighbors[B] = cost
        B.neighbors[A] = cost
```

This creates a **doubly-linked chain** per segment — the sweep path.

**Step 3: Cross-lap edges (horizontal links)**

```
for each new node N on lap k:
    for each existing node C where C.base_lap_index ∈ {k-1, k+1}:
        dist = euclidean_distance(N, C)
        if dist ≤ 1.5·w:
            if Bresenham line N→C is collision-free:
                N.neighbors[C] = dist
                C.neighbors[N] = dist
                N.cross_lap_neighbors.add(C)
                C.cross_lap_neighbors.add(N)
```

Cross-lap edges connect **adjacent laps only** (`|Δk| = 1`). Max distance is `1.5·w` (roughly `√(w² + w²) ≈ 1.06w` for diagonal neighbors, with headroom).

### Visual Summary

```
    Lap k-1          Lap k          Lap k+1

      ●                ●               ●
      │ same-lap       │               │ same-lap
      ●────────────────●───────────────●
      │   cross-lap    │   cross-lap   │
      ●                ●               ●
      │                │               │
      ●────────────────●               ●
                       │               │
                       ●───────────────●
                       │
                       ●

    ─── same-lap edge (vertical, within one segment)
    ─── cross-lap edge (horizontal, between k and k±1)
```

### prune() — Removing Inessential Nodes

After expansion, redundant nodes are removed. A node is **essential** if any of these hold:

| Condition | Why |
|-----------|-----|
| State is `CLOSED` | Already visited — structural |
| Is a retreat node | Needed for dead-end escape |
| `B(node, w)` touches unknown space | It's at the exploration frontier |
| Is a chain endpoint (`prev=None` or `next=None`) | Removing it would shorten the sweep |
| Is the sole cross-lap bridge to a chain endpoint on an adjacent lap | Removing it would disconnect that endpoint |

When removed, the chain is re-linked: `prev.next = next`, `next.prev = prev`, and a direct edge is added between them.

Non-essential **cross-lap edges** are also pruned unless they connect two chain endpoints or are the only bridge to one.

### revalidate_edges() — Handling Map Updates

Runs at 2 Hz on the viz timer. Re-checks every edge against the current occupancy grid:

```
for each edge (A, B):
    if Bresenham(A, B) now crosses an obstacle:
        remove from A.neighbors, B.neighbors
        clear same_lap_prev/next if applicable
        clear cross_lap_neighbors
```

### repair_connectivity() — Bridging Disconnections

If `revalidate_edges()` splits the graph, this finds connected components (DFS) and bridges them:

```
components = DFS connected components
anchor_component = the one containing the current robot node

for each other component:
    find shortest collision-free cross-lap edge to anchor
    (must be between laps k and k±1, dist ≤ 1.5·w)
    add the bridge edge
    merge into anchor
```

---

## 3. Waypoint Selection (Boustrophedon Sweep)

The `WaypointSelector` picks the next node using a **priority cascade**:

### Selection Priority

```
select_next(rcg, current_node):

    1. WALK FORWARD on same_lap_next chain
       → Follow .same_lap_next pointers, skip CLOSED, return first OPEN
       → If found: return it (no direction change)

    2. WALK REVERSE on same_lap_prev chain
       → Follow .same_lap_prev pointers, skip CLOSED, return first OPEN
       → If found: flip sweep direction, return it

    3. CROSS-LAP TRANSITION
       → Check cross_lap_neighbors for OPEN nodes on lap k-1 and k+1
       → Pick the lap with fewer remaining OPEN nodes (finish it first)
       → Flip sweep direction
       → If new direction is forward: pick node with lowest Y (start from bottom)
       → If new direction is reverse: pick node with highest Y (start from top)
       → If found: return it

    4. DEAD-END
       → No OPEN node reachable by any local edge
       → Return (None, is_dead_end=True)
```

### Boustrophedon Pattern

```
    Lap k           Lap k+1         Lap k+2

      ● start
      │ ↓ forward
      ●
      │
      ●
      │
      ● end of chain
       ╲
        ╲ cross-lap (flip direction)
         ╲
          ● ← enter at bottom
          │ ↑ reverse (now going up)
          ●
          │
          ●
          │
          ● end of chain
           ╲
            ╲ cross-lap (flip again)
             ╲
              ● ← enter at top
              │ ↓ forward again
              ●
              │
              ●
```

### Node State Update After Selection

Before moving to `next_id`, the **current node** is evaluated:

```
up_open   = same_lap_next exists and is OPEN
down_open = same_lap_prev exists and is OPEN

if both up_open AND down_open:
    keep current OPEN (robot is mid-chain, uncovered nodes on both sides)
else:
    close current node
```

### Link Node Injection

When **crossing laps** and the current node gets closed, there may be a gap — uncovered OPEN nodes more than `w` away on the same chain. A **link node** is injected to fill it:

```
if changing_lap AND current is now CLOSED:
    if up_open AND distance(current, next_above) > w:
        inject link node at current.y + w on same lap
        splice into chain: current ↔ link ↔ old_next
        mark link as retreat node

    if down_open AND distance(current, next_below) > w:
        inject link node at current.y - w on same lap
        splice into chain: old_prev ↔ link ↔ current
        mark link as retreat node
```

---

## 4. Dijkstra's Algorithm (Graph Path & Dead-End Escape)

### When It's Used

1. **Dead-end escape** — inner loop hits a dead-end, needs to reach the nearest OPEN node via the RCG graph.
2. **Repositioning** — outer loop finds no new frontier but OPEN nodes remain; robot navigates graph to reach them.
3. **Nearest OPEN by path** — `get_nearest_open_node_by_path()` finds the closest OPEN node by **graph distance**, not Euclidean distance.

### find_graph_path(from_id, to_id)

Standard Dijkstra on the RCG adjacency list. Edge weights are Euclidean distances.

```
function find_graph_path(from, to) → [node_ids]:
    dist  = {from: 0.0}
    prev  = {}
    pq    = min-heap [(0.0, from)]

    while pq not empty:
        d, nid = pop min from pq
        if nid == to: break
        if d > dist[nid]: continue     // stale entry

        for (adj, cost) in nodes[nid].neighbors:
            new_d = d + cost
            if new_d < dist.get(adj, ∞):
                dist[adj] = new_d
                prev[adj] = nid
                push (new_d, adj) to pq

    // Reconstruct
    path = []
    nid = to
    while nid ≠ from:
        path.prepend(nid)
        nid = prev[nid]
    return path                     // excludes from_id
```

If no path exists (disconnected graph), falls back to returning `[to_id]` directly.

### get_nearest_open_node_by_path(from_id)

Dijkstra variant — **early termination** at the first OPEN node popped:

```
function get_nearest_open_node_by_path(from):
    dist = {from: 0.0}
    pq   = [(0.0, from)]
    visited = {}

    while pq not empty:
        d, nid = pop min
        if nid in visited: continue
        visited.add(nid)

        if nid ≠ from AND nodes[nid].state == OPEN:
            return nodes[nid]        // ← first OPEN reached = closest by graph

        for (adj, cost) in nodes[nid].neighbors:
            new_d = d + cost
            if new_d < dist.get(adj, ∞):
                dist[adj] = new_d
                push (new_d, adj)

    return None   // no reachable OPEN node
```

### How the Robot Follows a Graph Path

Once Dijkstra returns a path `[n1, n2, ..., target]`, the robot walks it hop-by-hop:

```
for each node_id in path:
    node = rcg.nodes[node_id]
    success = navigator.go_to(node.x, node.y, prefer_direct=True)

    if success:
        mark edge covered (current → node)
        if node is OPEN → close it
        current = node_id
        proximity-close nearby OPEN nodes within w/2
    else:
        add node_id to skip-set, try next hop
```

Each hop is a physical navigation action (direct control or Nav2 fallback). Failed hops are skipped — the robot continues to the next waypoint in the path.

---

## 5. Velocity Planner & Motion Control

Every `navigator.go_to()` call goes through a two-tier system: **direct control** first, **Nav2 fallback** if it fails.

### Dispatch Logic

```
go_to(x, y, prefer_direct=True):
    if prefer_direct AND direct_enabled:
        result = _go_to_direct(x, y)
        if result == OK: return True
        if fallback_to_nav2:
            return _go_to_nav2(x, y)
        return False
    else:
        return _go_to_nav2(x, y)
```

### Direct Controller (P-Control + LiDAR Safety)

A simple proportional heading controller publishing `Twist` on `/cmd_vel` at 20 Hz.

**Heading:**

```
target_heading = atan2(dy, dx)
heading_err    = normalize(target_heading - robot_yaw)
ω = clamp(2.0 × heading_err,  -angular_speed .. +angular_speed)
```

**Forward speed:**

```
heading_factor = max(0, cos(heading_err))     ← slows down when not facing target
dist_factor    = min(1, distance / 0.25)      ← deceleration within 0.25m
v = linear_speed × heading_factor × dist_factor

if v < 0.015: v = 0     ← deadband: pure rotation when badly misaligned
```

**Arrival:** `distance to target < xy_tolerance (0.08m)` → stop, return success.

### LiDAR Safety Zones

The front ±35° arc of the `/scan` topic is monitored every cycle:

```
    ┌──────────────────────────────────────────┐
    │                                          │
    │   0.50m ─────── SLOW ZONE ──────────     │
    │     Linearly scale v by:                 │
    │     factor = (front_min - 0.30)          │
    │              / (0.50 - 0.30)             │
    │     clamped to [0.2, 1.0]                │
    │                                          │
    │   0.30m ─────── STOP ZONE ──────────     │
    │     v = 0  (rotation only)               │
    │     if blocked > 1.5 seconds → ABORT     │
    │                                          │
    │   0.22m ─────── EMERGENCY ──────────     │
    │     Immediate full stop → return FAIL    │
    │                                          │
    │   ████████████ OBSTACLE ████████████     │
    └──────────────────────────────────────────┘
```

### Grid Safety Check

Every 3 control cycles, Bresenham line from robot's current grid cell to the target grid cell is checked against the occupancy grid. If any cell is occupied or unknown → abort.

### Nav2 Fallback

If direct control fails (obstacle, timeout, grid safety), the same target is sent to Nav2 as a `NavigateToPose` action goal:

- Frame: `map`
- Timeout: 120s
- On rejection or timeout → cancel goal, return fail

### Recovery Manoeuvres

On navigation failure:

```
1. Backup 0.30m at 0.10 m/s (reverse on /cmd_vel)
2. Pick rotation direction:
   - Average LiDAR range on left (10°–85°) vs right (-85°–-10°)
   - Rotate 0.5 rad toward the side with more clearance
3. Add failed target to skip-set
4. If 5 consecutive failures → break inner loop
```

### Pseudocode: Direct Controller Loop

```
function _go_to_direct(target_x, target_y, timeout):
    blocked_since = None

    loop at 20Hz until timeout:
        rx, ry, ryaw = get_pose()
        dist = hypot(target_x - rx, target_y - ry)

        if dist ≤ 0.08m:
            stop()
            return OK

        heading_err = normalize(atan2(dy, dx) - ryaw)
        ω = clamp(2.0 * heading_err, ±1.0 rad/s)

        v = 0.20 * max(0, cos(heading_err)) * min(1, dist/0.25)
        if v < 0.015: v = 0

        front_min = min(scan.ranges in ±35°)

        if front_min < 0.22m:       → stop, return FAIL
        if front_min < 0.30m:
            v = 0
            if blocked_since is None: blocked_since = now
            if now - blocked_since > 1.5s: → stop, return FAIL
        elif front_min < 0.50m:
            v *= linear_scale(front_min, 0.30, 0.50)
            blocked_since = None
        else:
            blocked_since = None

        if cycle % 3 == 0:
            if NOT bresenham_clear(robot_grid, target_grid):
                stop, return FAIL

        publish(v, ω)
        sleep(50ms)

    stop()
    return FAIL (timeout)
```

---

## 6. Putting It All Together

```
 ┌──────────────────────────────────────────────────────────────┐
 │  1. SLAM provides occupancy grid                             │
 │                     ↓                                        │
 │  2. Sampler computes lap positions from sampling front       │
 │     Places FrontierSamples on each lap segment               │
 │                     ↓                                        │
 │  3. RCG.expand() creates nodes + same-lap chains +           │
 │     cross-lap edges (collision-checked)                      │
 │     RCG.prune() removes non-essential nodes                  │
 │                     ↓                                        │
 │  4. WaypointSelector.select_next():                          │
 │     Forward chain → reverse chain → cross-lap → dead-end     │
 │                     ↓                                        │
 │  5. Navigator moves robot to selected node                   │
 │                     ↓                                        │
 │  6. Node closed, edge coverage painted, retreat nodes updated│
 │                     ↓                                        │
 │  7. If dead-end → Dijkstra finds nearest OPEN → hop-by-hop  │
 │                     ↓                                        │
 │  8. If all OPEN exhausted → outer loop re-samples frontier   │
 │     New laps discovered → RCG grows → repeat from step 4     │
 │                     ↓                                        │
 │  9. No more frontier → coverage complete                     │
 └──────────────────────────────────────────────────────────────┘
```

