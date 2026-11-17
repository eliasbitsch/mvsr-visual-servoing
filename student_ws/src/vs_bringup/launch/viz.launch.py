"""foxglove_bridge for the Lichtblick web GUI, plus the relays it needs (see viz_relay.py)."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # use_sim_time: the bridge sends sim time, so the 3D panel looks up TF at sim stamps.
        # The clock is remapped to the 30 Hz copy, else the browser redraws ~500 times a second.
        Node(package='foxglove_bridge', executable='foxglove_bridge', output='screen',
             # 8766, not the usual 8765: that port is often taken by other local tools
             # Only what the layout shows: raw images (9 MB/s) and the 500 Hz model poses cost
             # most of the bridge's CPU and nobody looked at them
             parameters=[{'port': 8766, 'use_sim_time': True,
                          'topic_whitelist': [
                              '^/tf$', '^/tf_static$', '^/robot_description$', '^/vs/clock_viz$',
                              '^/vs/robot_description_viz$', '^/vs/scene_markers$',
                              '^/vs/debug_image/compressed$', '^/vs/features$',
                              '^/ibvs_node/.*', '^/ground_truth/pose_error$']}],
             remappings=[('/clock', '/vs/clock_viz')]),
        Node(package='vs_bringup', executable='viz_relay.py', output='screen'),
    ])
