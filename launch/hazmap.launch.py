"""
HazMap launch file — TurtleBot3 Burger + C* world + slam_toolbox + Nav2.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    os.environ['TURTLEBOT3_MODEL'] = 'burger'

    hazmap_dir    = get_package_share_directory('hazmap')
    c_star_dir    = get_package_share_directory('c_star_algorithm')
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz     = LaunchConfiguration('use_rviz', default='true')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time',
    )
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2',
    )

    hazmap_params = os.path.join(hazmap_dir, 'config', 'hazmap_params.yaml')
    nav2_params   = os.path.join(hazmap_dir, 'config', 'nav2_params.yaml')
    slam_params   = os.path.join(hazmap_dir, 'config', 'slam_toolbox_params.yaml')
    rviz_config   = os.path.join(hazmap_dir, 'config', 'hazmap.rviz')
    world_path    = os.path.join(c_star_dir, 'worlds', 'c_star_world.world')

    # ── Gazebo ─────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items(),
    )

    # ── TurtleBot3 robot state publisher ───────────────────────────
    tb3_rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # ── Spawn TurtleBot3 ───────────────────────────────────────────
    tb3_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '-2.0',
            'y_pose': '-0.5',
        }.items(),
    )

    # ── SLAM Toolbox (delay 5 s for Gazebo) ───────────────────────
    slam_toolbox = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    slam_params,
                    {'use_sim_time': True},
                ],
            ),
        ],
    )

    # ── Nav2 (delay 8 s for SLAM) ──────────────────────────────────
    nav2 = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': 'true',
                    'params_file': nav2_params,
                }.items(),
            ),
        ],
    )

    # ── HazMap coverage node (delay 20 s for full stack) ──────────
    hazmap_node = TimerAction(
        period=20.0,
        actions=[
            Node(
                package='hazmap',
                executable='hazmap_node',
                name='hazmap_node',
                output='screen',
                parameters=[
                    hazmap_params,
                    {'use_sim_time': use_sim_time},
                ],
            ),
        ],
    )

    # ── RViz2 (delay 22 s) ─────────────────────────────────────────
    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
    rviz = TimerAction(
        period=22.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                condition=IfCondition(use_rviz),
                arguments=rviz_args,
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_rviz,
        gazebo,
        tb3_rsp,
        tb3_spawn,
        slam_toolbox,
        nav2,
        hazmap_node,
        rviz,
    ])
