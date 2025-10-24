#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    addr = LaunchConfiguration('i2c_addr', default='0x48')
    bus = LaunchConfiguration('i2c_bus', default='1')
    ch = LaunchConfiguration('channel', default='0')
    vref = LaunchConfiguration('vref', default='5.0')
    rl = LaunchConfiguration('rl', default='10000.0')
    r0 = LaunchConfiguration('r0', default='10000.0')
    A = LaunchConfiguration('A', default='1000.0')
    B = LaunchConfiguration('B', default='1.5')
    rate = LaunchConfiguration('rate_hz', default='10.0')

    return LaunchDescription([
        DeclareLaunchArgument('i2c_addr', default_value=addr),
        DeclareLaunchArgument('i2c_bus', default_value=bus),
        DeclareLaunchArgument('channel', default_value=ch),
        DeclareLaunchArgument('vref', default_value=vref),
        DeclareLaunchArgument('rl', default_value=rl),
        DeclareLaunchArgument('r0', default_value=r0),
        DeclareLaunchArgument('A', default_value=A),
        DeclareLaunchArgument('B', default_value=B),
        DeclareLaunchArgument('rate_hz', default_value=rate),

        Node(
            package='app',
            executable='mq_ads1115_node',
            name='mq_ads1115_node',
            output='screen',
            parameters=[{
                'i2c_addr': addr,
                'i2c_bus': bus,
                'channel': ch,
                'vref': vref,
                'rl': rl,
                'r0': r0,
                'A': A,
                'B': B,
                'rate_hz': rate,
            }],
        ),
    ])


if __name__ == '__main__':
    generate_launch_description()
