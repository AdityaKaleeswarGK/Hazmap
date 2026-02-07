from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    my_pkg_dir = get_package_share_directory('hazmap')

    slam_params = os.path.join(
        my_pkg_dir, 'config', 'slam_toolbox_params.yaml'
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                slam_toolbox_dir,
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={
            'params_file': slam_params,
            'use_sim_time': 'false'
        }.items()
    )

    return LaunchDescription([
        slam_launch,
    ])
