import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    hazmap_dir = get_package_share_directory('hazmap')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    turtlebot3_model = LaunchConfiguration('model', default='waffle')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time',
    )
    declare_model = DeclareLaunchArgument(
        'model', default_value='waffle',
        description='TurtleBot3 model',
    )

    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', turtlebot3_model)

    # Config file paths
    slam_default_params = os.path.join(
        slam_toolbox_dir, 'config', 'mapper_params_online_async.yaml'
    )
    nav2_default_params = os.path.join(
        hazmap_dir, 'config', 'nav2_params.yaml'
    )
    hazmap_params = os.path.join(hazmap_dir, 'config', 'hazmap_params.yaml')
    rviz_config = os.path.join(hazmap_dir, 'config', 'hazmap.rviz')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch',
                         'turtlebot3_house.launch.py')
        ),
        launch_arguments={'x_pose': '-1.5', 'y_pose': '1.5'}.items(),
    )
    # ── SLAM Toolbox (delay 5 s for Gazebo) ─────────────────────
    slam = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_toolbox_dir, 'launch',
                                 'online_async_launch.py')
                ),
                launch_arguments={
                    'params_file': slam_default_params,
                    'use_sim_time': 'true',
                }.items(),
            ),
        ],
    )

    # ── Nav2 (delay 8 s for SLAM) ───────────────────────────────
    nav2 = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_dir, 'launch',
                                 'navigation_launch.py')
                ),
                launch_arguments={
                    'params_file': nav2_default_params,
                    'use_sim_time': 'true',
                    'autostart': 'true',
                }.items(),
            ),
        ],
    )

    # ── HazMap Coverage Node (delay 12 s for map + Nav2) ─────────
    hazmap_node = TimerAction(
        period=12.0,
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
    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
    rviz = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=rviz_args,
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_model,
        set_tb3_model,
        gazebo,
        slam,
        nav2,
        hazmap_node,
        rviz,
    ])

