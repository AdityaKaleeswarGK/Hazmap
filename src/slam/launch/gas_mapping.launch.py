import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def _setup(context):
    robot_name = LaunchConfiguration('robot_name', default='/').perform(context)
    map_topic = LaunchConfiguration('map_topic', default='map').perform(context)
    publish_prefix = LaunchConfiguration('publish_topic_prefix', default='gas_map').perform(context)
    map_frame = LaunchConfiguration('map_frame', default='map').perform(context)
    base_frame = LaunchConfiguration('base_frame', default='base_footprint').perform(context)
    kernel_radius = int(LaunchConfiguration('kernel_radius', default='1').perform(context))
    publish_hz = float(LaunchConfiguration('publish_hz', default='2.0').perform(context))
    ppm_threshold = float(LaunchConfiguration('ppm_threshold', default='20.0').perform(context))
    gases = LaunchConfiguration('gases', default="[{'name':'methane','topic':'gas/methane/ppm','max_concentration':10000.0},{'name':'lpg','topic':'gas/lpg/ppm','max_concentration':10000.0},{'name':'co','topic':'gas/co/ppm','max_concentration':10000.0},{'name':'air_quality','topic':'gas/air_quality/ppm','max_concentration':1000.0}]").perform(context)

    topic_prefix = '' if robot_name == '/' else f'/{robot_name}'

    node = Node(
        package='slam',
        executable='gas_mapper',
        name='gas_mapper',
        output='screen',
        parameters=[{
            'map_topic': f'{topic_prefix}/{map_topic}' if not map_topic.startswith('/') else map_topic,
            'publish_topic_prefix': f'{topic_prefix}/{publish_prefix}' if not publish_prefix.startswith('/') else publish_prefix,
            'map_frame': map_frame if map_frame.startswith(robot_name) or map_frame == 'map' else map_frame,
            'base_frame': base_frame if base_frame.startswith(robot_name) or base_frame == 'base_footprint' else base_frame,
            'kernel_radius': kernel_radius,
            'publish_hz': publish_hz,
            'ppm_threshold': ppm_threshold,
            'gases': gases,
        }]
    )

    return [GroupAction([PushRosNamespace(robot_name), node])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='/'),
        DeclareLaunchArgument('map_topic', default_value='map'),
        DeclareLaunchArgument('publish_topic_prefix', default_value='gas_map'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('kernel_radius', default_value='1'),
        DeclareLaunchArgument('publish_hz', default_value='2.0'),
        DeclareLaunchArgument('ppm_threshold', default_value='20.0'),
        DeclareLaunchArgument('gases', default_value="[{'name':'methane','topic':'gas/methane/ppm','max_concentration':10000.0},{'name':'lpg','topic':'gas/lpg/ppm','max_concentration':10000.0},{'name':'co','topic':'gas/co/ppm','max_concentration':10000.0},{'name':'air_quality','topic':'gas/air_quality/ppm','max_concentration':1000.0}]"),
        OpaqueFunction(function=_setup)
    ])
