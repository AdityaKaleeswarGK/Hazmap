from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    nav2_dir = get_package_share_directory('nav2_bringup')
    my_pkg_dir = get_package_share_directory('hazmap')

    nav2_params = os.path.join(
        my_pkg_dir, 'config', 'nav2_params.yaml'
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_dir,
                'launch',
                'navigation_launch.py'
            )
        ),
        launch_arguments={
            'params_file': nav2_params,
            'use_sim_time': 'false',
            'autostart': 'true'
        }.items()
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_dir,
                'launch',
                'rviz_launch.py'
            )
        )
    )

    return LaunchDescription([
        navigation_launch,
        rviz_launch
    ])
