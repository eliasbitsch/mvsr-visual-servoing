"""UR + wrist camera in Gazebo Harmonic for the visual servoing project.

    ros2 launch vs_sim vs_sim.launch.py
    ros2 launch vs_sim vs_sim.launch.py target:=red_can gazebo_gui:=false

Started controllers:
    joint_state_broadcaster               active
    scaled_joint_trajectory_controller    active   (used by go_to_start / capture_reference)
    forward_velocity_controller           inactive (activated by go_to_start)
"""

from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg = FindPackageShare("vs_sim")
    ur_type = LaunchConfiguration("ur_type")
    controllers_file = PathJoinSubstitution([pkg, "config", "controllers.yaml"])
    world_file = PathJoinSubstitution([pkg, "worlds", "vs_world.sdf"])
    gui = LaunchConfiguration("gazebo_gui").perform(context).lower() in ("1", "true", "yes")

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([pkg, "urdf", "ur_vs.urdf.xacro"]), " ",
        "name:=ur ",
        "ur_type:=", ur_type, " ",
        "safety_limits:=true ",
        "simulation_controllers:=", controllers_file, " ",
        "camera_rate:=", LaunchConfiguration("camera_rate"), " ",
        "camera_noise:=", LaunchConfiguration("camera_noise"),
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    gz_args = "-r -v 3 " if gui else "-s -r -v 3 --headless-rendering "
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": [gz_args, world_file]}.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-string", robot_description_content, "-name", "ur", "-allow_renaming", "true"],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        parameters=[{"config_file": PathJoinSubstitution([pkg, "config", "bridge.yaml"]),
                     "use_sim_time": True}],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    def spawner(name, *extra):
        return Node(package="controller_manager", executable="spawner",
                    arguments=[name, "-c", "/controller_manager", *extra], output="screen")

    ground_truth = Node(
        package="vs_sim", executable="ground_truth", name="ground_truth", output="screen",
        parameters=[{"use_sim_time": True, "active_target": LaunchConfiguration("target")}],
    )

    image_view = Node(
        package="rqt_image_view", executable="rqt_image_view", arguments=["/camera/image_raw"],
        condition=IfCondition(LaunchConfiguration("image_view")),
    )

    rviz = Node(
        package="rviz2", executable="rviz2", output="log",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("ur_description"), "rviz", "view_robot.rviz"])],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return [
        robot_state_publisher, gazebo, spawn_robot, bridge,
        spawner("joint_state_broadcaster"),
        spawner("scaled_joint_trajectory_controller"),
        spawner("forward_velocity_controller", "--inactive"),
        ground_truth, image_view, rviz,
    ]


def generate_launch_description():
    args = [
        DeclareLaunchArgument("ur_type", default_value="ur5e",
                              description="UR type (ur3e, ur5e, ur10e, ...). Start configs are tuned for ur5e."),
        DeclareLaunchArgument("target", default_value="aruco",
                              description="Active target for the ground-truth topics (see config/targets.yaml)."),
        DeclareLaunchArgument("gazebo_gui", default_value="true", description="Start the Gazebo GUI."),
        DeclareLaunchArgument("image_view", default_value="true", description="Show the camera image (rqt_image_view)."),
        DeclareLaunchArgument("rviz", default_value="false", description="Start RViz."),
        DeclareLaunchArgument("camera_rate", default_value="30", description="Camera frame rate [Hz]."),
        DeclareLaunchArgument("camera_noise", default_value="0.005", description="Gaussian image noise (stddev, 0..1)."),
    ]
    env = AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH",
                                    PathJoinSubstitution([FindPackageShare("vs_sim"), "models"]))
    return LaunchDescription(args + [env, OpaqueFunction(function=launch_setup)])
