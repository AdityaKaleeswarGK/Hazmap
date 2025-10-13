from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the path to the turtlebot3_gazebo package
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')

    # Path to the original turtlebot3_world.launch.py
    world_launch = os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_house.launch.py')

    # Include that launch file
    include_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(world_launch)
    )

    return LaunchDescription([
        include_world
    ])
