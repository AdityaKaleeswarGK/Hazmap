import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('jetauto_gazebo')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    slam_arg = DeclareLaunchArgument(
        'slam', default_value='true',
        description='Run SLAM (true) or use a saved map (false)',
    )
    map_arg = DeclareLaunchArgument(
        'map', default_value='',
        description='Full path to map yaml file (only used if slam:=false)',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to launch RViz2',
    )

    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'nav2_slam.rviz')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params,
        }.items(),
    )

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        condition=IfCondition(LaunchConfiguration('slam')),
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
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        slam_arg,
        map_arg,
        use_rviz_arg,
        nav2_bringup,
        slam_toolbox,
        rviz_node,
    ])
