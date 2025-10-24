from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 声明参数(declare parameters)
    lidar_frame = LaunchConfiguration('lidar_frame', default='lidar_frame')
    scan_raw = LaunchConfiguration('scan_raw', default='scan_raw')
    serial_port = LaunchConfiguration('serial_port', default='/dev/lidar')
    serial_baudrate = LaunchConfiguration('serial_baudrate', default='115200')
    scan_mode = LaunchConfiguration('scan_mode', default='Standard')
    channel_type = LaunchConfiguration('channel_type', default='serial')
    inverted = LaunchConfiguration('inverted', default='False')
    angle_compensate = LaunchConfiguration('angle_compensate', default='True')

    lidar_frame_arg = DeclareLaunchArgument('lidar_frame', default_value=lidar_frame)
    scan_raw_arg = DeclareLaunchArgument('scan_raw', default_value=scan_raw)
    serial_port_arg = DeclareLaunchArgument('serial_port', default_value=serial_port)
    serial_baudrate_arg = DeclareLaunchArgument('serial_baudrate', default_value=serial_baudrate)
    scan_mode_arg = DeclareLaunchArgument('scan_mode', default_value=scan_mode)
    channel_type_arg = DeclareLaunchArgument('channel_type', default_value=channel_type)
    inverted_arg = DeclareLaunchArgument('inverted', default_value=inverted)
    angle_compensate_arg = DeclareLaunchArgument('angle_compensate', default_value=angle_compensate)

    # 声明节点(declare node)
    a1_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        output='screen',
        parameters=[
            {
                'channel_type': channel_type,
                'serial_baudrate': serial_baudrate,
                'serial_port': serial_port,  # 统一映射为/dev/lidar(mapping all to /dev/lidar)
                'frame_id': lidar_frame,
                'inverted': inverted,
                'angle_compensate': angle_compensate,
                'scan_mode': scan_mode,
            }
        ],
        remappings=[('scan', scan_raw)]
    )

    return LaunchDescription([
        lidar_frame_arg,
        scan_raw_arg,
        serial_port_arg,
        serial_baudrate_arg,
        scan_mode_arg,
        channel_type_arg,
        inverted_arg,
        angle_compensate_arg,
        a1_node,
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
