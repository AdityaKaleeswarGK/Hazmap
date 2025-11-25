from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb/image_raw')
    model_path = LaunchConfiguration('model_path', default='')
    conf = LaunchConfiguration('conf', default='0.25')
    # Default params file installed with the package
    default_params = str(Path(get_package_share_directory('large_models')) / 'config' / 'yolo_infer.params.yaml')
    params_file = LaunchConfiguration('params_file', default=default_params)

    return LaunchDescription([
        Node(
            package='large_models',
            executable='yolo_infer',
            name='yolo_infer',
            output='screen',
            parameters=[
                params_file,
                {'camera_topic': camera_topic},
                {'model_path': model_path},
                {'conf': conf},
            ],
        )
    ])
