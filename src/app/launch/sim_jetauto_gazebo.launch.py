#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Arguments
    use_gui = LaunchConfiguration('use_gui', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='')  # empty world

    declare_use_gui = DeclareLaunchArgument('use_gui', default_value=use_gui)
    declare_use_rviz = DeclareLaunchArgument('use_rviz', default_value=use_rviz)
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value=use_sim_time)
    declare_world = DeclareLaunchArgument('world', default_value=world)

    # Ensure description package resolves correctly (this repo uses need_compile)
    set_need_compile = SetEnvironmentVariable(name='need_compile', value=os.environ.get('need_compile', 'True'))

    # Gazebo Classic
    gazebo_pkg = get_package_share_directory('gazebo_ros')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')),
        launch_arguments={'verbose': 'true', 'pause': 'false', 'world': world}.items(),
    )

    # Robot description (URDF) and state publisher + optional RViz
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

    # Gas field simulator
    gas_field_sim = Node(
        package='app',
        executable='gas_field_sim',
        name='gas_field_sim',
        output='screen',
        parameters=[{
            'world_frame': 'odom',
            'base_frame': 'base_link',
            'sources': [
                {'x': 2.0, 'y': 0.0, 'z': 0.0, 'strength': 5e5},
            ],
            'falloff': 2.0,
            'rate_hz': 10.0,
        }],
        condition=IfCondition(use_sim_time),
    )

    return LaunchDescription([
        declare_use_gui,
        declare_use_rviz,
        declare_use_sim_time,
        declare_world,
        set_need_compile,
        gazebo_launch,
        robot_desc_launch,
        spawn_entity,
        gas_field_sim,
    ])


if __name__ == '__main__':
    generate_launch_description()
