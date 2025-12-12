from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('prometheus_frontier_explorer')

    # Arguments
    slam_arg = DeclareLaunchArgument(
        'slam', default_value='True',
        description='Whether to launch SLAM'
    )
    nav_arg = DeclareLaunchArgument(
        'nav', default_value='True',
        description='Whether to launch Navigation'
    )

    # Includes
    # We use the existing launch files in this package for modularity
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'slam.launch.py')),
        condition=IfCondition(LaunchConfiguration('slam'))
    )

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'nav2_bringup.launch.py')),
        condition=IfCondition(LaunchConfiguration('nav'))
    )

    # Nodes
    visualizer_node = Node(
        package='prometheus_frontier_explorer',
        executable='frontier_visualizer',
        name='frontier_visualizer',
        output='screen'
    )

    explorer_node = Node(
        package='prometheus_frontier_explorer',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen'
    )

    return LaunchDescription([
        slam_arg,
        nav_arg,
        slam_launch,
        nav_launch,
        visualizer_node,
        explorer_node
    ])
