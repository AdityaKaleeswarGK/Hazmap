# Hazpatrol — Surveillance Path Planning with C*

**Pardon the naming;it still refers to the previous name Hazmap didnt update it everywhere cause im just lazy to do it**

Autonomous **surveillance** path planning for TurtleBot3 on **ROS 2 Humble**, built around a modified **C\*** algorithm. Rather than exhaustively lawnmower-covering every square metre, the robot *surveys* an unknown environment: it concentrates its observation effort on cluttered, feature-rich regions and samples open empty space sparsely. It builds a Reachability Connectivity Graph (RCG) incrementally as the map grows and sweeps the navigable area in boustrophedon laps.

The intended deployment is gas / hazard monitoring. The patrol is designed to run inside a **GADEN** gas-dispersion simulation; when auto-start is enabled the node waits for GADEN's `/odor_value` and `/wind_value` services before beginning a sweep. (GADEN currently provides the operating environment — the gas plume the robot patrols through — but the planner does not yet steer goal selection from live concentration/wind readings; surveillance behaviour comes from the adaptive sampling below.)



---

## Algorithm Overview

The implementation follows the C\* coverage algorithm with several practical modifications, re-tuned so the sweep behaves like a surveillance patrol rather than exhaustive floor coverage.

### 1. Progressive Sampling (surveillance-weighted)

At each step, the robot samples the newly revealed free space within its detection radius (`rd`). Samples are placed on parallel laps spaced `w` metres apart, oriented along the chosen sweep axis. Only samples adjacent to unknown space or obstacles (i.e., on the frontier) are retained — this keeps the graph sparse and focused on regions that still need visiting.

**Adaptive frontier density** is what makes this a surveillance sweep rather than a lawnmower one. A sample is classified as "open" when the nearest obstacle is at least `density_open_distance` away; in those open, empty regions only a `density_keep_floor` fraction of lap samples are kept (default 0.40 → ~60% dropped). Cluttered, feature-rich regions stay densely sampled. The robot therefore spends its time observing where there is structure to inspect and skims across empty floor, instead of dutifully ticking off every node in an open room. Set `density_keep_floor = 1.0` to fall back to uniform coverage sampling.

### 2. Reachability Connectivity Graph (RCG)

Frontier samples become nodes in the RCG. Each node carries:
- Its world position `(x, y)` and which lap it belongs to
- State: **OPEN** (not yet visited) or **CLOSED** (covered)
- Directional neighbor lists: `up`, `down` (same lap), `left`, `right` (adjacent laps)

The graph is built incrementally as the robot moves:
- **Expand** — add new nodes from the latest frontier samples and connect them to existing nodes (same-lap neighbors and cross-lap neighbors within `√2 · w`).
- **Prune** — remove inessential nodes. A node is *essential* if it borders unknown space, is an end node of its lap, or is the only cross-lap bridge to an end node on an adjacent lap. Everything else is pruned to keep the graph lean.

### 3. Goal Selection

At every step the selector picks the next OPEN node to visit. Priority order: left-lap neighbor → up-lap neighbor → down-lap neighbor → right-lap neighbor. When no adjacent neighbor is available (dead end), A\* on the RCG finds the nearest reachable OPEN node.

Two practical modifications over the original algorithm:

- **Local commitment** (`commit_threshold`): a nearby same-lap neighbor wins over a distant candidate unless the distant one offers more than `1/commit_threshold` times the coverage gain. This keeps the robot sweeping locally rather than jumping across the map.
- **Path-cost penalty** (`path_blocked_penalty`): if the straight line to a candidate is collision-blocked, its effective cost is multiplied by this factor. Prevents the selector from picking a geometrically close node that is physically far around a wall.

### 4. Coverage Tracking

The occupancy grid manager tracks which free cells have been visited. A cell is marked covered when the robot passes within `rc` metres of it. Coverage is tracked both at waypoints and along traversed segments (interpolated between waypoints) so narrow corridors are counted correctly.

### 5. Termination

The patrol stops under any of three conditions:
- **Target reached**: observed area exceeds 95% of known free space.
- **Stagnation**: neither observed area nor discovered free area has grown meaningfully in the last 8 arrivals.
- **Diminishing returns**: area-observed-per-metre-travelled over a rolling window falls below threshold while observed area is already above 80%.

A safety-net frontier-cluster search runs when no lap samples remain but the map still has unexplored openings — this catches narrow passages and off-axis gaps the lap sampler misses.

> Throughout, "coverage" in the code and parameter names refers to *area the robot has observed/passed within `rc` of* — the surveillance footprint, not a guarantee of exhaustive floor coverage.

---

## Architecture

```
                        ┌────────────────────────────────────────────────┐
                        │                  hazpatrolNode                 │
                        │           (hazmap_core/hazmap_node.py)         │
                        │                                                │
   /map ───────────────►   ┌──────────────────────┐                      │
   /odom ──────────────►   │ OccupancyGridManager │                      │
   /scan ──────────────►   │                      │  frontier queries    │
                        │  │  • free/unknown/occ  │◄─────────────────┐   │
                        │  │  • coverage tracking │                  │   │
                        │  │  • obstacle distance │                  │   │
                        │  └──────────┬───────────┘                  │   │
                        │             │ grid state                   │   │
                        │  ┌──────────▼───────────┐                  │   │
                        │  │  ProgressiveSampler  │                  │   │
                        │  │                      │  (x,y,lap,pos)   │   │
                        │  │  • lap grid at w m   ├──────────────►   │   │
                        │  │  • frontier samples  │                  │   │
                        │  │  • coverage mask     │          ┌───────┴───┴──────┐
                        │  └──────────────────────┘          │       RCG        │
                        │                                    │                  │
                        │                                    │  • expand()      │
                        │                                    │  • prune()       │
                        │                                    │  • A* search     │
                        │                                    │  • OPEN/CLOSED   │
                        │                                    └───────┬──────────┘
                        │                                            │ next node
                        │  ┌─────────────────────────────────────────▼──────┐  │
                        │  │               GoalSelector                     │  │
                        │  │                                                │  │
                        │  │  SelectGoalNode: Left→Up→Down→Right            │  │
                        │  │  local commitment  │  path-cost penalty        │  │
                        │  │  dead-end escape via RCG A*                    │  │
                        │  └─────────────────────────┬──────────────────────┘  │
                        │                            │ (x, y) goal             │
                        │  ┌─────────────────────────▼──────────────────────┐  │
                        │  │                  Navigator                     │  │
                        │  │                                                │  │
                        │  │  Nav2 NavigateToPose / NavigateThroughPoses    │  │
                        │  │  optional: direct velocity control (/cmd_vel)  │  │
                        │  └─────────────────────────┬──────────────────────┘  │
                        │                            │                         │
                        └────────────────────────────┼─────────────────────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │    TurtleBot3 +     │
                                          │    Nav2 + SLAM      │
                                          └─────────────────────┘
```

**Data flow per step:**
1. `OccupancyGridManager` ingests the latest `/map` and tracks covered cells.
2. `ProgressiveSampler` places frontier samples on parallel laps within the robot's detection radius.
3. `RCG.expand()` adds new nodes; `RCG.prune()` removes inessential ones.
4. `GoalSelector.select_goal_node()` picks the next OPEN node (left-lap preferred; falls back to A\* on dead end).
5. `Navigator` drives the robot there via Nav2; on arrival `close_nearby_nodes()` marks covered nodes CLOSED.
6. Repeat until termination (coverage target, stagnation, or diminishing returns).

---

## Stack

- **ROS 2 Humble** on Ubuntu 22.04
- **TurtleBot3 Burger** (example platform; any differential-drive robot works)
- **Nav2** for global path following
- **SLAM Toolbox** for online mapping
- **GADEN** gas-dispersion simulator for the operating environment (provides `/odor_value` and `/wind_value`; launched separately)
- Navigation: Nav2 `NavigateToPose` / `NavigateThroughPoses`, with optional direct velocity control for short collision-free hops

---

## Project Structure

```
hazmap/
├── config/
│   ├── hazmap_params.yaml          # All tunable parameters
│   ├── nav2_params.yaml            # Nav2 planner/controller params
│   └── slam_toolbox_params.yaml    # SLAM Toolbox params
├── launch/
│   └── hazmap.launch.py            # Launches Gazebo, SLAM, Nav2, HazMap, RViz
├── hazmap/
│   └── hazmap_core/
│       ├── hazmap_node.py          # Main ROS 2 node — orchestrates everything
│       ├── occupancy_grid_manager.py  # Grid queries, frontier detection, coverage stats
│       ├── progressive_sampling.py    # Lap-based frontier sampling
│       ├── rcg.py                     # RCG: expand, prune, A*
│       ├── goal_selection.py          # SelectGoalNode, UpdateState, dead-end escape
│       └── navigator.py               # Nav2 action client + direct velocity control
├── package.xml
├── setup.py
└── requirements.txt
```

---

## Installation

```bash
cd ~/ros2_ws/src
git clone <repo-url> hazmap

pip install -r hazmap/requirements.txt

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select hazmap --symlink-install
source install/setup.bash
```

---

## Usage

```bash
export TURTLEBOT3_MODEL=burger

# Launch Gazebo + SLAM Toolbox + Nav2 + HazMap + RViz
ros2 launch hazmap hazmap.launch.py

# In another terminal — start the surveillance sweep
ros2 service call /hazmap/start_coverage std_srvs/srv/Trigger

# Stop early (saves visit log)
ros2 service call /hazmap/stop_coverage std_srvs/srv/Trigger
```

The launch file brings up Gazebo (in the `c_star_world`), SLAM, Nav2, the HazMap node and RViz — it does **not** start GADEN. Run your GADEN gas-dispersion launch separately. With `auto_start_coverage: true`, HazMap waits until `/map`, odometry, the Nav2 `navigate_to_pose` server, **and** GADEN's `/odor_value` + `/wind_value` services are all available, then begins the sweep on its own — no manual `start_coverage` call needed. With auto-start off (the default), the GADEN gate does not apply and you trigger the sweep manually.

---

## Key Parameters (`config/hazmap_params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w` | 0.50 m | Lap spacing |
| `rc` | 0.30 m | Coverage radius per waypoint |
| `rd` | 3.0 m | Detection / sampling radius |
| `sweep_direction` | `"x"` | `"x"` = vertical laps, `"y"` = horizontal |
| `density_keep_floor` | 0.40 | Fraction of lap samples kept in open space (1.0 = uniform coverage; 0.0 = strip all open-area nodes) |
| `density_open_distance` | 1.5 m | Distance from nearest obstacle for a sample to count as "open" |
| `auto_start_coverage` | `false` | Auto-begin the sweep once `/map`, odom, Nav2 and GADEN services are ready |
| `use_hybrid_navigation` | `false` | Enable direct velocity control for short hops |
| `commit_threshold` | 0.30 | Local-commitment bias (lower = stronger local preference) |
| `path_blocked_penalty` | 3.0 | Cost multiplier for collision-blocked candidates |
| `nav_step_timeout_s` | 50.0 | Per-step Nav2 timeout (seconds) |
| `prune_enable` | `true` | Prune covered OPEN nodes to reduce overlap |
| `eff_stop_enable` | `true` | Stop on diminishing-returns criterion |
| `eff_stop_min_coverage` | 80.0 % | Minimum coverage before efficiency stop kicks in |

---

## ROS 2 Interface

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/hazmap/rcg_nodes` | `MarkerArray` | RCG nodes (green=OPEN, red=CLOSED, yellow=current) |
| `/hazmap/rcg_edges` | `MarkerArray` | RCG edges (yellow=same-lap, cyan=cross-lap) |
| `/hazmap/current_goal` | `Marker` | Current navigation target |
| `/hazmap/coverage_path` | `Path` | Sequence of visited waypoints |
| `/hazmap/robot_trajectory` | `Path` | Dense real-time trajectory |
| `/hazmap/laps` | `MarkerArray` | Colour-coded lap lines |
| `/hazmap/frontier_points` | `MarkerArray` | Current frontier samples |
| `/hazmap/observation_quality` | `OccupancyGrid` | Coverage density grid |

### Subscribed Topics

| Topic | Type |
|-------|------|
| `/map` | `OccupancyGrid` |
| `/odom` | `Odometry` |
| `/scan` | `LaserScan` |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/hazmap/start_coverage` | `Trigger` | Begin autonomous coverage |
| `/hazmap/stop_coverage` | `Trigger` | Stop and save visit log |

---

## Output

On completion a timestamped CSV is saved to `~/ros2_ws/results/YYYYMMDD_HHMMSS/visit_log.csv` with one row per navigation step: goal position, navigation result, and coverage statistics at arrival.
