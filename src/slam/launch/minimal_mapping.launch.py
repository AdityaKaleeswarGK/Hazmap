import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Minimal mapping bringup: Lidar + IMU (calib+filter) + Controller/EKF + slam_toolbox
# Excludes cameras, ASR, large models, and app behaviors to keep CPU/memory low.

def _setup(context):
    compiled = os.environ.get('need_compile', 'True')
    robot_name = LaunchConfiguration('robot_name', default='/').perform(context)
    sim = LaunchConfiguration('sim', default='false').perform(context)

    if compiled == 'True':
        periph_pkg = get_package_share_directory('peripherals')
        slam_pkg = get_package_share_directory('slam')
        controller_pkg = get_package_share_directory('controller')
    else:
        periph_pkg = '/home/ubuntu/ros2_ws/src/peripherals'
        slam_pkg = '/home/ubuntu/ros2_ws/src/slam'
        controller_pkg = '/home/ubuntu/ros2_ws/src/controller'

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(periph_pkg, 'launch/lidar.launch.py')),
        launch_arguments={'sim': sim}.items()
    )

    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(periph_pkg, 'launch/imu_filter.launch.py'))
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(controller_pkg, 'launch/controller.launch.py')),
        launch_arguments={'sim': sim}.items()
    )

    slam_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_pkg, 'launch/include/slam_base.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'scan_topic': '/scan',
            'enable_save': 'true'
        }.items()
    )

    return [
        lidar_launch,
        imu_launch,
        controller_launch,
        TimerAction(period=5.0, actions=[slam_base])
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='/'),
        DeclareLaunchArgument('sim', default_value='false'),
        OpaqueFunction(function=_setup)
    ])
