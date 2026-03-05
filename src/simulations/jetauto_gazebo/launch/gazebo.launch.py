import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    gazebo_pkg_dir = get_package_share_directory('jetauto_gazebo')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    description_pkg_dir = get_package_share_directory('jetauto_description')

    os.environ['MACHINE_TYPE'] = 'JetAuto'
    os.environ['LIDAR_TYPE'] = 'A1'
    os.environ['DEPTH_CAMERA_TYPE'] = 'AstraProPlus'
    os.environ['need_compile'] = 'True'

    model_path = os.path.join(description_pkg_dir, '..')
    gazebo_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    if gazebo_model_path:
        os.environ['GAZEBO_MODEL_PATH'] = model_path + ':' + gazebo_model_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] = model_path

    set_gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH', os.environ['GAZEBO_MODEL_PATH']
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(gazebo_pkg_dir, 'worlds', 'empty.world'),
        description='Full path to the Gazebo world file',
    )
    x_arg = DeclareLaunchArgument('x', default_value='0.0', description='Spawn X position')
    y_arg = DeclareLaunchArgument('y', default_value='2.0', description='Spawn Y position')
    z_arg = DeclareLaunchArgument('z', default_value='0.1', description='Spawn Z position')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0', description='Spawn yaw angle')

    xacro_file = os.path.join(gazebo_pkg_dir, 'urdf', 'jetauto_gazebo.urdf.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': LaunchConfiguration('world')}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_jetauto',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'jetauto',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
    )

    return LaunchDescription([
        set_gazebo_model_path,
        world_arg,
        x_arg,
        y_arg,
        z_arg,
        yaw_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity,
    ])
