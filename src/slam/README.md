# SLAM & Mapping Guide

This document explains how to run 2D mapping with `slam_toolbox` in this workspace, optionally auto‑save the map as image (PNG/PDF), and switch to alternative RGB‑D SLAM (`rtabmap`).

---
## 1. Components
| Component | Purpose | Key Topics / Frames |
|-----------|---------|---------------------|
| Lidar driver (peripherals) | Publishes laser scans | `<ns>/scan` |
| Controller + EKF (`robot_localization`) | Fuses wheel odom + IMU | `odom`, TF `odom->base_footprint` |
| IMU broadcaster | Publishes IMU + TF | `imu`, `imu_link` |
| slam_toolbox (sync node) | 2D pose graph & occupancy grid | `map`, TF `map->odom` |
| Map auto saver (new) | Saves `.pgm/.yaml` and converts to `.png/.pdf` | Services `/slam_toolbox/save_map`, `/slam_toolbox/serialize_map` |

Frame tree (expected): `map -> odom -> base_footprint -> (lidar_frame, imu_link, ...)`

---
## 2. Prerequisites
1. Workspace built with ROS 2 (e.g. Humble/Foxy equivalent used by the project).
2. Packages installed: `slam_toolbox`, `robot_localization`, `usb_cam`/depth stack if using RGB‑D.
3. Environment variables used by existing launch files:
   - `need_compile` = `True` (installed) or `False` (source layout) 
   - `HOST`, `MASTER` (namespace logic) – if unset they default to `/`.
4. Optional image conversion: install ImageMagick (`sudo apt install imagemagick`).

---
## 3. Build
```bash
colcon build --packages-select slam peripherals controller servo_controller
. install/setup.bash
```
(Windows: Use WSL or a ROS 2 Windows environment; PowerShell sourcing differs.)

---
## 4. Basic 2D Mapping (slam_toolbox)
```bash
export need_compile=True
ros2 launch slam mapping_bringup.launch.py robot_name:=jetbot sim:=false
```
Open RViz (if not already):
```bash
ros2 launch slam rviz_slam.launch.py
```
Set `Fixed Frame` to `map` and add Map + LaserScan.

Stop mapping with Ctrl+C when satisfied.

---
## 5. Automatic Map Saving (PNG/PDF)
Enable the auto saver in the same launch:
```bash
ros2 launch slam mapping_bringup.launch.py \
  robot_name:=jetbot \
  map_auto_save:=true \
  map_save_path:=/home/ubuntu/maps \
  map_save_delay:=30.0 \
  map_save_png:=true \
  map_save_pdf:=true
```
Results (example):
```
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS.pgm
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS.yaml
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS.png
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS.pdf
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS_graph.posegraph
/home/ubuntu/maps/slam_map_YYYYMMDD_HHMMSS_graph.data
```
Arguments:
- `map_auto_save` (true|false)
- `map_save_delay` seconds before first (and only) save
- `map_save_path` output directory (created if missing)
- `map_base_name` stem (default `slam_map`)
- `map_save_png`, `map_save_pdf` toggles

---
## 6. Manual Map Save (on demand)
```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: '/home/ubuntu/maps/manual_map'}"
```
Produces `manual_map.pgm` + `manual_map.yaml`.

Serialize pose graph (optional):
```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/ubuntu/maps/manual_graph'}"
```

---
## 7. Converting Maps (If auto saver disabled)
```bash
magick convert manual_map.pgm manual_map.png
magick convert manual_map.pgm manual_map.pdf
```
Or Python:
```python
from PIL import Image; Image.open('manual_map.pgm').save('manual_map.png')
```

---
## 8. Switching to RGB‑D (rtabmap) (Optional)
If you have a depth camera and want loop closures with vision:
1. Launch depth camera (set `use_depth_camera:=true` in `include/robot.launch.py` or your own bringup).
2. Use `rtabmap_slam.launch.py` instead of `mapping_bringup.launch.py` (or adapt). Example:
```bash
ros2 launch slam rtabmap_slam.launch.py
```
3. Topics remapped inside `rtabmap.launch.py` already expect `/depth_cam/...` naming.

Note: For pure 2D occupancy, `slam_toolbox` is lighter and recommended.

---
## 9. Parameter Quick Reference (slam/config/slam.yaml)
| Param | Effect |
|-------|--------|
| `resolution` | Occupancy grid cell size (m) |
| `minimum_travel_distance` / `minimum_travel_heading` | Keyframe insertion thresholds |
| `loop_search_maximum_distance` | Loop closure search radius |
| `map_update_interval` | Map publish frequency |
| `max_laser_range` | Rasterization upper bound |

Adjust these if map looks sparse (reduce thresholds) or CPU high (increase thresholds / lower frequency).

---
## 10. Common Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| Map warps early | EKF not stable yet | Increase launch delay before SLAM start (TimerAction) |
| No PNG/PDF | ImageMagick missing | Install `imagemagick` or enable `map_save_png/pdf` and rebuild |
| Map rotated / drift | Wrong TF or wheel params | Verify `odom->base_footprint` and encoder calibration |
| Service timeout saving | SLAM busy | Increase delay or reduce scan rate/throttle |

---
## 11. Reusing a Saved Map (Localization Mode)
For localization-only (after building a static map):
1. Choose the best saved map `X.pgm` and `X.yaml`.
2. Run `nav2_map_server` or switch slam_toolbox `mode: localization` in `slam.yaml` and set `map_file_name: /path/to/X`.

---
## 12. Next Steps
- Integrate Nav2 (global/local planners) with produced map.
- Periodic map snapshots (extend auto saver: set `once:=False`).
- Add adaptive scan filtering if noise is high.

---
## 13. Minimal Troubleshooting Commands
```bash
ros2 topic hz /scan
ros2 topic echo /tf | head -n 20
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo odom base_footprint
```

---
## 14. Support
Ensure you rebuilt after adding new nodes:
```bash
colcon build --packages-select slam
. install/setup.bash
```
Check the node list:
```bash
ros2 node list | grep map_auto_saver
```
If absent, verify launch arguments and that `map_auto_save:=true` was passed.

---
Happy mapping.
