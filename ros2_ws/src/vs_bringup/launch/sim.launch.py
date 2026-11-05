"""Gazebo cell: GoFa + Robotiq 2F-85 + eye-in-hand D435, controllers and topic bridge."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def spawner(name, *extra, condition=None):
    return Node(package='controller_manager', executable='spawner',
                arguments=[name, '--controller-manager-timeout', '60', *extra],
                parameters=[{'use_sim_time': True}], condition=condition, output='screen')


def generate_launch_description():
    pkg = get_package_share_directory('vs_bringup')
    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')
    use_gripper = LaunchConfiguration('use_gripper')

    # Gazebo resolves package:// mesh URIs by looking for the package folder in this path.
    mesh_roots = {os.path.dirname(get_package_share_directory(p))
                  for p in ('gofa_description', 'robotiq_description', 'realsense2_description')}

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(pkg, 'urdf', 'vs_cell.urdf.xacro'),
                 ' use_gripper:=', use_gripper]),
        value_type=str)

    gz_flags = PythonExpression(["'-r -v 2 ' if '", gui, "' == 'true' else '-r -v 2 -s --headless-rendering '"])
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']),
        launch_arguments={'gz_args': [gz_flags, world], 'on_exit_shutdown': 'true'}.items())

    create = Node(package='ros_gz_sim', executable='create', output='screen',
                  arguments=['-topic', 'robot_description', '-name', 'vs_cell'])

    controllers = [
        spawner('joint_state_broadcaster'),
        spawner('arm_trajectory_controller'),
        spawner('arm_velocity_controller', '--inactive'),
        spawner('gripper_controller', condition=IfCondition(use_gripper)),
    ]

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI'),
        DeclareLaunchArgument('use_gripper', default_value='true'),
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds', 'vs_world.sdf')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.pathsep.join(sorted(mesh_roots))),
        gazebo,
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': os.path.join(pkg, 'config', 'bridge.yaml'),
                          'use_sim_time': True}]),
        create,
        RegisterEventHandler(OnProcessExit(target_action=create, on_exit=controllers)),
    ])
