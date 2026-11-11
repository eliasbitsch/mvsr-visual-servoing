#!/usr/bin/env python3
"""Republish /robot_description with package:// mesh URIs for the Lichtblick 3D panel.

The lab URDF references meshes as file:///opt/ros/jazzy/share/<pkg>/...; a browser may not load
local files, while Lichtblick fetches package:// URIs through foxglove_bridge.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


def main():
    rclpy.init()
    node = Node('viz_robot_description')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(String, '/vs/robot_description_viz', latched)
    node.create_subscription(
        String, '/robot_description',
        lambda m: pub.publish(String(data=m.data.replace('file:///opt/ros/jazzy/share/', 'package://'))),
        latched)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
