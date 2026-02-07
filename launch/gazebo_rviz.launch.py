from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    world_launch = os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_world.launch.py')
    include_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(world_launch)
    )
    return LaunchDescription([
        include_world
    ])
