#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Arguments
    use_gui = LaunchConfiguration('use_gui', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_use_gui = DeclareLaunchArgument('use_gui', default_value=use_gui)
    declare_use_rviz = DeclareLaunchArgument('use_rviz', default_value=use_rviz)
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value=use_sim_time)

    # Ensure env flag used by repo is set
    set_need_compile = SetEnvironmentVariable(name='need_compile', value=os.environ.get('need_compile', 'True'))

    # Gazebo Classic
    gazebo_pkg = get_package_share_directory('gazebo_ros')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={'verbose': 'true', 'pause': 'false'}.items(),
    )

    # Robot description and RViz (optional)
    jetauto_desc_pkg = get_package_share_directory('jetauto_description')
    robot_desc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(jetauto_desc_pkg, 'launch', 'robot_description.launch.py')),
        launch_arguments={
            'use_gui': use_gui,
            'use_rviz': use_rviz,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # Spawn entity in Gazebo from robot_description
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'jetauto'],
        output='screen',
    )

    # Multi-gas field simulator
    gas_sim = Node(
        package='app',
        executable='gas_field_sim_multi',
        name='gas_field_sim_multi',
        output='screen',
        parameters=[{
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'rate_hz': 10.0,
            'gases': [
                {'name': 'methane', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
                 'sources': [{'x': 2.0, 'y': 0.0, 'z': 0.0, 'strength': 5e5}]},
                {'name': 'lpg', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
                 'sources': [{'x': -2.0, 'y': 1.0, 'z': 0.0, 'strength': 5e5}]},
                {'name': 'co', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
                 'sources': [{'x': 0.0, 'y': 2.0, 'z': 0.0, 'strength': 5e5}]},
                {'name': 'air_quality', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
                 'sources': [{'x': -1.5, 'y': -1.0, 'z': 0.0, 'strength': 5e5}]},
            ],
        }]
    )

    # Optional: run app-level gas mapper to visualize without SLAM
    app_mapper = Node(
        package='app',
        executable='gas_mapper',
        name='gas_mapper',
        output='screen',
        parameters=[{
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'resolution': 0.2,
            'width': 200,
            'height': 200,
            'origin_x': -20.0,
            'origin_y': -20.0,
            'decay_rate': 0.0,
            'pub_rate': 2.0,
            'ppm_threshold': 20.0,
            'topics': [
                {'name': 'methane', 'topic': 'gas/methane/ppm'},
                {'name': 'lpg', 'topic': 'gas/lpg/ppm'},
                {'name': 'co', 'topic': 'gas/co/ppm'},
                {'name': 'air_quality', 'topic': 'gas/air_quality/ppm'},
            ],
            'topic_max_ppm': [
                {'name': 'methane', 'max_ppm': 10000.0},
                {'name': 'lpg', 'max_ppm': 10000.0},
                {'name': 'co', 'max_ppm': 10000.0},
                {'name': 'air_quality', 'max_ppm': 1000.0},
            ],
        }]
    )

    return LaunchDescription([
        declare_use_gui,
        declare_use_rviz,
        declare_use_sim_time,
        set_need_compile,
        gazebo_launch,
        robot_desc_launch,
        spawn_entity,
        gas_sim,
        app_mapper,
    ])


if __name__ == '__main__':
    generate_launch_description()
