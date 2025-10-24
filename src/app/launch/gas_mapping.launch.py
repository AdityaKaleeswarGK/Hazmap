#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Ensure env flag used by this repo exists
    set_need_compile = SetEnvironmentVariable(name='need_compile', value=os.environ.get('need_compile', 'True'))

    rate_hz = LaunchConfiguration('rate_hz', default='10.0')
    declare_rate = DeclareLaunchArgument('rate_hz', default_value=rate_hz)

    # Multi-sensor ADS1115 reader
    ads_node = Node(
        package='app',
        executable='mq_ads1115_multi',
        name='mq_ads1115_multi',
        output='screen',
        parameters=[{
            'i2c_addr': 0x48,
            'i2c_bus': 1,
            'rate_hz': rate_hz,
            'sensors': [
                {'name': 'methane', 'channel': 0, 'vref': 5.0, 'rl': 10000.0, 'r0': 12000.0, 'A': 1000.0, 'B': 1.5},
                {'name': 'lpg', 'channel': 1, 'vref': 5.0, 'rl': 10000.0, 'r0': 12000.0, 'A': 1000.0, 'B': 1.5},
                {'name': 'co', 'channel': 2, 'vref': 5.0, 'rl': 10000.0, 'r0': 12000.0, 'A': 1000.0, 'B': 1.5},
                {'name': 'air_quality', 'channel': 3, 'vref': 5.0, 'rl': 10000.0, 'r0': 12000.0, 'A': 1000.0, 'B': 1.5},
            ],
        }]
    )

    # Gas mapper node
    mapper = Node(
        package='app',
        executable='gas_mapper',
        name='gas_mapper',
        output='screen',
        parameters=[{
            'world_frame': 'map',
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
        set_need_compile,
        declare_rate,
        ads_node,
        mapper,
    ])


if __name__ == '__main__':
    generate_launch_description()
