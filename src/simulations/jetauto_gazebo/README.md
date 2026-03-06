# jetauto_gazebo

ROS 2 Humble package to simulate the JetAuto robot in Gazebo Classic 11 with SLAM and Nav2 support.

## Dependencies

```
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-slam-toolbox \
  ros-humble-nav2-bringup ros-humble-teleop-twist-keyboard \
  ros-humble-joint-state-publisher ros-humble-robot-state-publisher \
  ros-humble-xacro
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-up-to jetauto_gazebo
source install/setup.bash
```

## URDF Config

The Gazebo wrapper xacro (`urdf/jetauto_gazebo.urdf.xacro`) includes the original `jetauto_description` URDF and adds:

| Plugin | Topic | Purpose |
|--------|-------|---------|
| `planar_move` | `/cmd_vel`, `/odom` | Mecanum holonomic drive |
| `ray_sensor` | `/scan` | A1 LiDAR (front 180, 12m range) |
| `imu_sensor` | `/imu/data` | IMU |
| `camera` | `/depth_camera/*` | Depth camera |

Environment variables set by the launch file:
- `MACHINE_TYPE=JetAuto`
- `LIDAR_TYPE=A1 (only the front 180 degree of the lidar reading is sent to the scan topic)`
- `DEPTH_CAMERA_TYPE=AstraProPlus`

## Commands

### Launch Gazebo

```bash
ros2 launch jetauto_gazebo gazebo.launch.py
```

With test world:

```bash
ros2 launch jetauto_gazebo gazebo.launch.py \
  world:=$(ros2 pkg prefix jetauto_gazebo)/share/jetauto_gazebo/worlds/test_world.world
```

### SLAM + Nav2

```bash
ros2 launch jetauto_gazebo nav2.launch.py
```

### Teleop

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Save Map

```bash
ros2 run nav2_map_server map_saver_cli -f ~/maps/jetauto_map
```

### SLAM Only (no Nav2)

```bash
ros2 launch jetauto_gazebo slam.launch.py
```

## Nav2 Config

Nav2 params (`config/nav2_params.yaml`) are tuned for mecanum drive:
- `robot_radius: 0.17`
- `max_vel_y: 0.2` (lateral strafing enabled)
- `robot_model_type: OmniMotionModel`

## Set Goal

In RViz2 use the **2D Goal Pose** button, or via terminal:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 3.0, y: 3.0}, orientation: {w: 1.0}}}}"
```
