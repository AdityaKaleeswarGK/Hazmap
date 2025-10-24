import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import PushRosNamespace, Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Unified bringup for sensors + EKF + 2D SLAM (slam_toolbox) to build a map.
# Added optional auto map saving (PNG/PDF) using map_auto_saver node.
# Usage (example):
#   export need_compile=True  # or False if using source layout like other files in this repo
#   ros2 launch slam mapping_bringup.launch.py robot_name:=jetbot slam_method:=slam_toolbox enable_save:=true
# Key frames: map -> odom -> base_footprint -> imu_link / lidar_frame
# Topics used:
#   - <robot_ns>/scan     (Laser scan)
#   - <robot_ns>/odom     (Filtered odom from EKF)
#   - <robot_ns>/imu      (IMU orientation + angular velocity)
# Output map topics:
#   - <robot_ns>/map
#   - <robot_ns>/map_metadata

def _setup(context):
    compiled = os.environ.get('need_compile', 'True')
    slam_method = LaunchConfiguration('slam_method', default='slam_toolbox').perform(context)
    sim = LaunchConfiguration('sim', default='false').perform(context)
    robot_name = LaunchConfiguration('robot_name', default=os.environ.get('HOST', '/')).perform(context)
    master_name = LaunchConfiguration('master_name', default=os.environ.get('MASTER', '/')).perform(context)
    enable_save = LaunchConfiguration('enable_save', default='true').perform(context)
    map_auto_save = LaunchConfiguration('map_auto_save', default='false').perform(context)
    map_save_path = LaunchConfiguration('map_save_path', default='/tmp').perform(context)
    map_save_delay = LaunchConfiguration('map_save_delay', default='10.0').perform(context)
    map_save_pdf = LaunchConfiguration('map_save_pdf', default='false').perform(context)
    map_save_png = LaunchConfiguration('map_save_png', default='true').perform(context)
    base_map_name = LaunchConfiguration('map_base_name', default='slam_map').perform(context)
    # Gas mapping controls
    enable_gas_mapping = LaunchConfiguration('enable_gas_mapping', default='false').perform(context)
    gas_topic = LaunchConfiguration('gas_topic', default='gas_sensor/concentration').perform(context)
    gas_publish_topic = LaunchConfiguration('gas_publish_topic', default='gas_map').perform(context)
    gas_kernel_radius = LaunchConfiguration('gas_kernel_radius', default='1').perform(context)
    gas_max_ppm = LaunchConfiguration('gas_max_ppm', default='1000.0').perform(context)

    frame_prefix = '' if robot_name == '/' else f"{robot_name}/"
    topic_prefix = '' if robot_name == '/' else f"/{robot_name}"
    use_sim_time = 'true' if sim == 'true' else 'false'

    map_frame = f"{frame_prefix}map" if robot_name == master_name else f"{master_name}/map"
    odom_frame = f"{frame_prefix}odom"
    base_frame = f"{frame_prefix}base_footprint"
    scan_topic = f"{topic_prefix}/scan"
    map_topic = f"{topic_prefix}/map"

    if compiled == 'True':
        slam_pkg = get_package_share_directory('slam')
    else:
        slam_pkg = '/home/ubuntu/ros2_ws/src/slam'

    # Reuse existing robot bringup (controller + lidar + imu + joystick, depth camera optional)
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_pkg, 'launch/include/robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name,
            'use_joy': 'false',  # can enable externally
            'use_depth_camera': 'false'
        }.items()
    )

    # Core slam_toolbox base (already parameterized)
    slam_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_pkg, 'launch/include/slam_base.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'scan_topic': scan_topic,
            'enable_save': enable_save,
        }.items(),
    )

    auto_saver_node = None
    if map_auto_save == 'true':
        auto_saver_node = Node(
            package='slam',
            executable='map_auto_saver',
            name='map_auto_saver',
            output='screen',
            parameters=[{
                'save_path': map_save_path,
                'base_name': base_map_name,
                'delay_sec': float(map_save_delay),
                'make_pdf': (map_save_pdf == 'true'),
                'make_png': (map_save_png == 'true'),
                'once': True,
                'save_pose_graph': True,
            }]
        )

    gas_mapper_node = None
    if enable_gas_mapping == 'true':
        gas_mapper_node = Node(
            package='slam',
            executable='gas_mapper',
            name='gas_mapper',
            output='screen',
            parameters=[{
                'map_topic': map_topic,
                'gas_topic': gas_topic if gas_topic.startswith('/') else f'{topic_prefix}/{gas_topic}',
                'publish_topic': gas_publish_topic if gas_publish_topic.startswith('/') else f'{topic_prefix}/{gas_publish_topic}',
                'map_frame': map_frame,
                'base_frame': base_frame,
                'kernel_radius': int(gas_kernel_radius),
                'max_concentration': float(gas_max_ppm),
                'publish_hz': 2.0,
            }]
        )

    bringup = GroupAction([
        PushRosNamespace(robot_name),
        robot_launch,
        TimerAction(period=5.0, actions=[slam_base_launch]),  # wait for EKF + lidar
    ] + ([auto_saver_node] if auto_saver_node else []) + ([gas_mapper_node] if gas_mapper_node else []))

    return [bringup]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='/'),
        DeclareLaunchArgument('master_name', default_value='/'),
        DeclareLaunchArgument('slam_method', default_value='slam_toolbox'),
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('enable_save', default_value='true'),
        DeclareLaunchArgument('map_auto_save', default_value='false'),
        DeclareLaunchArgument('map_save_path', default_value='/tmp'),
        DeclareLaunchArgument('map_save_delay', default_value='10.0'),
        DeclareLaunchArgument('map_save_pdf', default_value='false'),
        DeclareLaunchArgument('map_save_png', default_value='true'),
        DeclareLaunchArgument('map_base_name', default_value='slam_map'),
        DeclareLaunchArgument('enable_gas_mapping', default_value='false'),
        DeclareLaunchArgument('gas_topic', default_value='gas_sensor/concentration'),
        DeclareLaunchArgument('gas_publish_topic', default_value='gas_map'),
        DeclareLaunchArgument('gas_kernel_radius', default_value='1'),
        DeclareLaunchArgument('gas_max_ppm', default_value='1000.0'),
        OpaqueFunction(function=_setup)
    ])
