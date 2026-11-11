"""FM perception + IBVS controller against the lab simulation (started separately, see README).

Example:
  ros2 launch vs_bringup servo.launch.py detector:=yolo_world prompt:="blue box"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('vs_bringup'), 'config', 'servo.yaml')
    arg = LaunchConfiguration

    return LaunchDescription([
        DeclareLaunchArgument('detector', default_value='grounding_dino'),
        DeclareLaunchArgument('prompt', default_value='red can'),
        DeclareLaunchArgument('tracker', default_value='csrt'),
        DeclareLaunchArgument('refiner', default_value='none'),
        DeclareLaunchArgument('detector_period', default_value='1.0'),
        DeclareLaunchArgument('detect_on_loss', default_value='false',
                              description='Run the detector only while no track exists'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='Start servoing as soon as the target is seen'),

        # Perception needs the venv interpreter (numpy 2, OpenCV 4.12, PyTorch), see Dockerfile.fm
        Node(package='vs_perception', executable='perception_node', output='screen',
             prefix=os.environ.get('VS_PYTHON', ''),
             # Thread pools default to one thread per core and spin while idle (~4 cores
             # burnt on a 20-thread laptop); the heavy work runs on the GPU anyway.
             additional_env={'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                             'OPENBLAS_NUM_THREADS': '2'},
             parameters=[params, {'detector': arg('detector'), 'prompt': arg('prompt'),
                                  'tracker': arg('tracker'), 'refiner': arg('refiner'),
                                  'detector_period': arg('detector_period'),
                                  'detect_on_loss': arg('detect_on_loss')}]),
        Node(package='vs_control', executable='ibvs_node', output='screen',
             parameters=[params, {'auto_start': arg('auto_start')}]),
    ])
