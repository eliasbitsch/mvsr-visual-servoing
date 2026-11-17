#!/usr/bin/env python3
"""Relays for the Lichtblick web GUI.

- /robot_description -> /vs/robot_description_viz with package:// mesh URIs: the lab URDF
  references meshes as file:///opt/ros/jazzy/share/<pkg>/..., which a browser may not load,
  while Lichtblick fetches package:// URIs through foxglove_bridge.
- /clock (~500 Hz) -> /vs/clock_viz at 30 Hz: the bridge forwards every clock tick to the
  browser as a time update, and Lichtblick redraws on each one.
- /vs/scene_markers: the objects of the lab world as markers, shapes and colours from the model
  SDFs and poses from the world file (static objects; no ground truth involved).
"""
import math
import os
import xml.etree.ElementTree as ET

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

CLOCK_RATE = 30.0


def _floats(text):
    return [float(x) for x in text.split()]


def _pose(xyz, yaw):
    return Pose(position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def scene_markers(share):
    """One marker per visual of every model included in the world (planar poses only)."""
    world = ET.parse(os.path.join(share, 'worlds', 'vs_world.sdf')).getroot()
    out = MarkerArray()
    for inc in world.iter('include'):
        name = inc.findtext('name')
        x, y, z, _, _, yaw = _floats(inc.findtext('pose', '0 0 0 0 0 0'))
        sdf = os.path.join(share, 'models', inc.findtext('uri').replace('model://', ''), 'model.sdf')
        for visual in ET.parse(sdf).getroot().iter('visual'):
            vx, vy, vz = _floats(visual.findtext('pose', '0 0 0 0 0 0'))[:3]
            m = Marker(ns=name, id=len(out.markers), action=Marker.ADD, frame_locked=True)
            m.header.frame_id = 'world'
            geom = visual.find('geometry')
            if geom.find('box') is not None:
                m.type = Marker.CUBE
                m.scale.x, m.scale.y, m.scale.z = _floats(geom.findtext('box/size'))
            elif geom.find('cylinder') is not None:
                m.type = Marker.CYLINDER
                r = float(geom.findtext('cylinder/radius'))
                m.scale.x = m.scale.y = 2 * r
                m.scale.z = float(geom.findtext('cylinder/length'))
            elif geom.find('sphere') is not None:
                m.type = Marker.SPHERE
                m.scale.x = m.scale.y = m.scale.z = 2 * float(geom.findtext('sphere/radius'))
            else:
                continue
            c, s_ = math.cos(yaw), math.sin(yaw)
            m.pose = _pose((x + c * vx - s_ * vy, y + s_ * vx + c * vy, z + vz), yaw)
            r, g, b, a = _floats(visual.findtext('material/diffuse', '0.7 0.7 0.7 1'))
            m.color = ColorRGBA(r=r, g=g, b=b, a=a)
            out.markers.append(m)
    return out


def main():
    rclpy.init()
    node = Node('viz_relay')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    urdf_pub = node.create_publisher(String, '/vs/robot_description_viz', latched)
    node.create_subscription(
        String, '/robot_description',
        lambda m: urdf_pub.publish(String(data=m.data.replace('file:///opt/ros/jazzy/share/',
                                                              'package://'))),
        latched)

    marker_pub = node.create_publisher(MarkerArray, '/vs/scene_markers', latched)
    marker_pub.publish(scene_markers(get_package_share_directory('vs_sim')))

    clock_pub = node.create_publisher(Clock, '/vs/clock_viz', 10)
    latest = {}
    node.create_subscription(Clock, '/clock', lambda m: latest.update(msg=m),
                             qos_profile_sensor_data)
    node.create_timer(1.0 / CLOCK_RATE,
                      lambda: 'msg' in latest and clock_pub.publish(latest.pop('msg')))
    rclpy.spin(node)


if __name__ == '__main__':
    main()
