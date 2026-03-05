import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('jetauto_gazebo')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to launch RViz2',
    )

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            os.path.join(pkg_dir, 'config', 'slam_params.yaml'),
            {'use_sim_time': True},
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'nav2_slam.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        use_rviz_arg,
        slam_toolbox,
        rviz_node,
    ])
