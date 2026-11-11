"""foxglove_bridge for the Lichtblick web GUI, plus the URDF relay the 3D panel needs."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='foxglove_bridge', executable='foxglove_bridge', output='screen',
             # 8766, not the usual 8765: that port is often taken by other local tools
             # use_sim_time: the bridge sends sim time, so the 3D panel looks up TF at sim stamps
             parameters=[{'port': 8766, 'use_sim_time': True}]),
        Node(package='vs_bringup', executable='viz_robot_description.py', output='screen'),
    ])
