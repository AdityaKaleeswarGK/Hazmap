# HazMap — Hazard Mapping & Coverage Path Planning

> Autonomous hazard mapping system for TurtleBot3 on **ROS 2 Humble**.  
> Combines coverage path planning, simulated environmental sensors, and an optional YOLO-based visual detection pipeline to build consolidated hazard impact maps.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
  - [Sensors](#sensors)
  - [Visual Detection Classes](#visual-detection-classes)
  - [Navigation Parameters](#navigation-parameters)
- [Usage](#usage)
  - [Launch](#launch)
  - [Start / Stop Coverage](#start--stop-coverage)
  - [Enabling the CV Detection Pipeline](#enabling-the-cv-detection-pipeline)
- [ROS 2 Topics & Services](#ros-2-topics--services)
- [Output & Results](#output--results)
- [Architecture](#architecture)

---

## Overview

HazMap drives a TurtleBot3 (Waffle) through an unknown environment using a progressive frontier-based coverage algorithm built on a **Reachability Connectivity Graph (RCG)**. As the robot moves, environmental sensors are sampled and an optional camera-based YOLO detector localises visual hazards. All data is fused into a **consolidated hazard impact map** with priority-weighted overlays and saved as publication-ready figures.

## Features

| Category | Details |
|---|---|
| **Coverage Planning** | Progressive frontier sampling, boustrophedon sweep, RCG expansion/pruning, dead-end escape via graph search |
| **Hybrid Navigation** | Direct LiDAR-safe local planner with automatic Nav2 fallback |
| **Environmental Sensors** | Simulated CO, CO₂, Methane, O₂ with Gaussian-splash heatmaps |
| **Visual Detection** | Optional YOLO pipeline — RGB+Depth → 3D map-frame localisation with EMA tracking and duplicate suppression |
| **Consolidated Map** | Priority-weighted fusion of all sensor and detection data into a single hazard impact map |
| **Visualisation** | Real-time RViz markers (RCG nodes/edges, sensor cylinders, detection pins) + saved PNG result figures |

---

## Project Structure

```
hazmap/
├── config/
│   ├── hazmap_params.yaml          # All tunable parameters
│   ├── hazmap.rviz                 # RViz display config
│   └── nav2_params.yaml            # Nav2 planner/controller params
│
├── launch/
│   └── hazmap.launch.py            # Launches Gazebo, SLAM, Nav2, HazMap, RViz
│
├── model/                          # Place YOLO model here to enable CV pipeline
│   └── .gitkeep
│
├── hazmap/
│   ├── __init__.py
│   │
│   ├── hazmap_core/                # Core coverage planning modules
│   │   ├── __init__.py
│   │   ├── hazmap_node.py          # Main ROS 2 node — orchestrates everything
│   │   ├── map_manager.py          # Occupancy grid, frontier masks, coverage masks
│   │   ├── sampling.py             # Progressive frontier sampling
│   │   ├── rcg.py                  # Reachability Connectivity Graph
│   │   ├── waypoint_selector.py    # Boustrophedon sweep waypoint selection
│   │   ├── navigator.py            # Hybrid navigation (direct + Nav2 fallback)
│   │   └── utils.py                # Grid/world conversions, collision checks
│   │
│   └── pipeline/                   # Sensor + CV detection pipeline
│       ├── __init__.py
│       ├── sensor_manager.py       # Sensor simulation, marker publishing, heatmap saving
│       └── detection_manager.py    # YOLO detection, 3D localisation, object tracking
│
├── resource/
│   └── hazmap
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
│
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **ROS 2 Humble** (Ubuntu 22.04)
- **TurtleBot3 packages** (`turtlebot3_gazebo`, `turtlebot3_description`)
- **Nav2** (`nav2_bringup`)
- **SLAM Toolbox** (`slam_toolbox`)
- **Python 3.10+**
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

## Configuration

All parameters live in **`config/hazmap_params.yaml`** under the `hazmap_node` namespace.

### Sensors

Four environmental sensors are defined by default, each with a name, unit, and priority (lower = higher importance in the consolidated map):

| Sensor  | Unit | Priority |
|---------|------|----------|
| CO      | ppm  | 1        |
| CO₂     | ppm  | 2        |
| Methane | ppm  | 3        |
| O₂      | %    | 4        |

```yaml
sensor_names:      ["co",  "co2", "methane", "o2"]
sensor_units:      ["ppm", "ppm", "ppm",     "%"]
sensor_priorities: [1,     2,     3,         4]
```

### Visual Detection Classes

Five YOLO detection classes are configured with severity priorities (lower = more severe):

| Class | Description | Priority |
|-------|-------------|----------|
| Fire in shrouded environments | `fire` | 1 (most severe) |
| Smoke | `smoke` | 2 |
| Spills in shrouded environments | `spills_shrouded` | 3 |
| Translucent spills | `translucent_spills` | 4 |
| Distinctly coloured spills | `coloured_spills` | 5 (least severe) |

```yaml
detection_class_names: [
  "coloured_spills",
  "fire",
  "spills_shrouded",
  "smoke",
  "translucent_spills"
]
detection_priorities: [5, 1, 3, 2, 4]
```

### Navigation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w` | 0.75 | Lap spacing (metres) |
| `delta` | 1 | Sample spacing multiplier |
| `sweep_direction` | `"x"` | `"x"` = vertical laps, `"y"` = horizontal |
| `use_hybrid_navigation` | `true` | Enable direct LiDAR-safe planner |
| `direct_nav_max_distance` | 0.70 | Max edge length for direct mode (m) |
| `direct_nav_linear_speed` | 0.20 | Forward speed for direct mode (m/s) |
| `direct_nav_angular_speed` | 1.00 | Turning speed (rad/s) |
| `direct_nav_min_clearance` | 0.18 | Min obstacle clearance for direct mode (m) |
| `direct_nav_fallback_to_nav2` | `true` | Fall back to Nav2 on direct failure |

See `config/hazmap_params.yaml` for the full parameter list.

---

## Usage

### Launch

```bash
# Set TurtleBot3 model (default: waffle)
export TURTLEBOT3_MODEL=waffle

# Launch everything: Gazebo + SLAM + Nav2 + HazMap + RViz
ros2 launch hazmap hazmap.launch.py
```


### Start / Stop Coverage

```bash
# Start autonomous coverage
ros2 service call /hazmap/start_coverage std_srvs/srv/Trigger

# Stop coverage (the robot will halt and save results)
ros2 service call /hazmap/stop_coverage std_srvs/srv/Trigger
```

Coverage also saves results automatically when the frontier is exhausted, or on `Ctrl+C`.

### Enabling the CV Detection Pipeline

The visual detection pipeline is **conditionally enabled**. To activate it:

1. Place a YOLO model file (`.pt`, `.onnx`, or `.engine`) in the `model/` directory:
   ```bash
   cp /path/to/your/yolov8n.pt ~/ros2_ws/src/hazmap/model/
   ```
2. Install the optional dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```
3. Rebuild and relaunch.

If the `model/` directory is empty or `ultralytics` is not installed, the CV pipeline is silently disabled and the rest of the system runs normally.

The detection pipeline:
- Subscribes to synchronised RGB + Depth camera topics
- Runs YOLO inference on each frame
- Projects detections to 3D using camera intrinsics and depth
- Transforms 3D points from camera frame to map frame via TF2
- Tracks unique objects with EMA position updates and duplicate suppression
- Publishes coloured sphere + text markers to RViz on `/hazmap/detections`

---

## ROS 2 Topics & Services

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/hazmap/rcg_nodes` | `MarkerArray` | RCG node spheres (green=open, red=closed, yellow=current) |
| `/hazmap/rcg_edges` | `MarkerArray` | RCG edge lines (yellow=same-lap, cyan=cross-lap) |
| `/hazmap/current_goal` | `Marker` | Current navigation goal sphere |
| `/hazmap/coverage_path` | `Path` | Sequence of visited waypoints |
| `/hazmap/robot_trajectory` | `Path` | Dense real-time robot trajectory |
| `/hazmap/laps` | `MarkerArray` | Coloured lap lines |
| `/hazmap/frontier_points` | `MarkerArray` | Current frontier sample points |
| `/hazmap/detections` | `MarkerArray` | Detected object pins + labels (if CV enabled) |
| `/hazmap/<sensor>_markers` | `MarkerArray` | Per-sensor reading cylinders (e.g. `/hazmap/co_markers`) |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/map` | `OccupancyGrid` | SLAM occupancy grid |
| `/odom` | `Odometry` | Robot odometry |
| `/camera/color/image_raw` | `Image` | RGB camera (if CV enabled) |
| `/camera/depth/image_raw` | `Image` | Depth camera (if CV enabled) |
| `/camera/color/camera_info` | `CameraInfo` | Camera intrinsics (if CV enabled) |
| `/scan` | `LaserScan` | LiDAR for direct navigation safety |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/hazmap/start_coverage` | `Trigger` | Begin autonomous coverage |
| `/hazmap/stop_coverage` | `Trigger` | Stop coverage and save results |

---

## Output & Results

When coverage completes (or is stopped), results are saved to a timestamped directory:

```
~/ros2_ws/results/YYYYMMDD_HHMMSS/
├── co_heatmap.png                  # CO concentration heatmap
├── co2_heatmap.png                 # CO₂ concentration heatmap
├── methane_heatmap.png             # Methane concentration heatmap
├── o2_heatmap.png                  # O₂ concentration heatmap
├── detection_pins.png              # Visual detection pin map (if CV enabled)
├── consolidated_impact.png         # Priority-weighted combined hazard map
└── coverage_connectivity_graph.png # RCG graph with executed path overlay
```

### Consolidated Hazard Map

The consolidated map fuses all sensor and detection data:

- Each sensor grid is normalised to [0, 1] and weighted by `1 / priority`
- Detection data is projected as a Gaussian splash impact grid and weighted by average detection severity
- The combined map includes danger contour lines at 50%, 70%, and 90% levels
- Detection pins are overlaid as coloured markers with splash radius circles

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     HazMapNode                          │
│  (hazmap_core/hazmap_node.py)                           │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ MapManager  │  │ Progressive  │  │     RCG       │   │
│  │             │  │  Sampler     │  │               │   │
│  └──────┬───── ┘  └──────┬───────┘  └───────┬───────┘   │
│         │                │                   │          │
│  ┌──────┴────────────────┴───────────────────┴───────┐  │
│  │              WaypointSelector                     │  │
│  └─────────────────────┬─────────────────────────────┘  │
│                        │                                │
│  ┌─────────────────────┴─────────────────────────────┐  │
│  │     Navigator (Direct + Nav2 Fallback)            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │   SensorManager     │  │   DetectionManager       │  │
│  │ (pipeline/)         │  │ (pipeline/) [optional]   │  │
│  │                     │  │                          │  │
│  │  CO, CO₂, Methane,  │  │  YOLO → Depth → TF2      │  │
│  │  O₂ simulation      │  │  → Map-frame pins        │  │
│  └──────────┬──────────┘  └────────────┬─────────────┘  │
│             │                          │                │
│             └──────────┬───────────────┘                │
│                        ▼                                │
│            ┌───────────────────────┐                    │
│            │  Consolidated Hazard  │                    │
│            │     Impact Map        │                    │
│            └───────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

1. **MapManager** maintains the SLAM occupancy grid and coverage state.
2. **ProgressiveSampler** generates frontier samples at the boundary of explored space.
3. **RCG** organises samples into a navigable graph with lap connectivity.
4. **WaypointSelector** picks the next target using boustrophedon sweep logic.
5. **Navigator** executes motion — fast direct control for short edges, Nav2 for longer ones.
6. **SensorManager** samples simulated environmental sensors at 4 Hz along the trajectory.
7. **DetectionManager** (optional) runs YOLO on camera frames and tracks 3D detections.
8. On completion, all data is fused into a consolidated hazard impact map.

---

