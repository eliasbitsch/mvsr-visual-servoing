"""Full pipeline: simulation + FM perception + IBVS controller.

Example:
  ros2 launch vs_bringup servo.launch.py detector:=yolo_world prompt:="blue cylinder"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('vs_bringup')
    params = os.path.join(pkg, 'config', 'servo.yaml')
    arg = LaunchConfiguration

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true', description='Start Gazebo as well'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('detector', default_value='grounding_dino'),
        DeclareLaunchArgument('prompt', default_value='red box'),
        DeclareLaunchArgument('tracker', default_value='csrt'),
        DeclareLaunchArgument('refiner', default_value='none'),
        DeclareLaunchArgument('detector_period', default_value='1.0'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='Start servoing as soon as the target is seen'),

        IncludeLaunchDescription(os.path.join(pkg, 'launch', 'sim.launch.py'),
                                 launch_arguments={'gui': arg('gui')}.items(),
                                 condition=IfCondition(arg('sim'))),
        # Perception needs the venv interpreter (numpy 2, OpenCV 4.12, PyTorch), see Dockerfile
        Node(package='vs_perception', executable='perception_node', output='screen',
             prefix=os.environ.get('VS_PYTHON', ''),
             parameters=[params, {'detector': arg('detector'), 'prompt': arg('prompt'),
                                  'tracker': arg('tracker'), 'refiner': arg('refiner'),
                                  'detector_period': arg('detector_period')}]),
        Node(package='vs_control', executable='ibvs_node', output='screen',
             parameters=[params, {'auto_start': arg('auto_start')}]),
    ])
