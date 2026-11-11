"""Typed commands for the robot: "fahre über die rote Dose", "move above the blue box", "stop".

Subscribes /vs/command (std_msgs/String) and reports on /vs/command_status. For a goto command
it stops servoing, sets the perception prompt, waits until the target is seen and then starts
the IBVS controller, which drives the camera above the object. If the target is not in view
within a few seconds, the arm first moves to a fixed overview pose (overview_joints) from which
the whole object group is visible, and the search continues there.

  ros2 run vs_perception command_node
  ros2 run vs_perception vs_cmd            # type commands interactively
"""
import os
import queue
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from vs_interfaces.msg import ObjectFeatures

from vs_perception.command_parser import GOTO, STOP, parse_command


class CommandNode(Node):

    def __init__(self):
        super().__init__('command_node')
        params = {
            'use_model': True,
            'url': os.environ.get('VS_VLM_URL', 'http://localhost:11434/v1'),
            'model': os.environ.get('VS_VLM_MODEL', 'qwen3-vl-gpu:4b'),
            'perception_node': 'perception_node',
            'ibvs_node': 'ibvs_node',
            'find_timeout': 30.0,
            # Seconds to look from the current pose before moving to the overview pose
            'quick_find_timeout': 5.0,
            # Joint angles of the overview pose (order of arm_joints), empty = no overview
            'overview_joints': [0.0],
            'arm_joints': [''],
            'trajectory_action': '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            'max_joint_speed': 0.6,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in params}
        self.overview = list(self.p['overview_joints'])
        if len(self.overview) != len(self.p['arm_joints']):
            self.overview = []

        self.status_pub = self.create_publisher(String, '/vs/command_status', 10)
        self.create_subscription(String, '/vs/command', self.on_command, 10)
        self.create_subscription(ObjectFeatures, '/vs/features', self.on_features, 10)
        self.joints = {}
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.trajectory = ActionClient(self, FollowJointTrajectory, self.p['trajectory_action'])
        self.set_params = self.create_client(
            SetParameters, f'/{self.p["perception_node"]}/set_parameters')
        self.start_srv = self.create_client(Trigger, f'/{self.p["ibvs_node"]}/start')
        self.stop_srv = self.create_client(Trigger, f'/{self.p["ibvs_node"]}/stop')

        self.features = None
        self.commands = queue.Queue()
        self.cancel = threading.Event()
        threading.Thread(target=self.worker, daemon=True).start()
        self.report(f'ready, model {self.p["model"]}' if self.p['use_model'] else 'ready, rules only')

    def report(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def on_features(self, msg):
        self.features = msg

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def on_command(self, msg):
        # A new command interrupts the one still waiting for its target
        self.cancel.set()
        self.commands.put(msg.data.strip())

    def call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{client.srv_name} not available')
        future = client.call_async(request)
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f'{client.srv_name} did not answer')
        return future.result()

    def stop_servoing(self):
        self.call(self.stop_srv, Trigger.Request())

    def set_prompt(self, prompt):
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=prompt)
        req = SetParameters.Request(parameters=[Parameter(name='prompt', value=value)])
        result = self.call(self.set_params, req).results[0]
        if not result.successful:
            raise RuntimeError(f'prompt rejected: {result.reason}')

    def wait_future(self, future, timeout, what):
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end and not self.cancel.is_set():
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f'{what}: no answer' if not self.cancel.is_set() else 'cancelled')
        return future.result()

    def move_to_overview(self):
        """Joint trajectory to the overview pose. Servoing must be stopped, which switches the
        arm to the trajectory controller; the switch is asynchronous, hence the retries."""
        names = list(self.p['arm_joints'])
        if not self.trajectory.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'{self.p["trajectory_action"]} not available')
        q_now = [self.joints.get(n, q) for n, q in zip(names, self.overview)]
        duration = max(2.0, max(abs(a - b) for a, b in zip(self.overview, q_now))
                       / self.p['max_joint_speed'])
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        sec = int(duration)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=self.overview, velocities=[0.0] * len(names),
            time_from_start=Duration(sec=sec, nanosec=int((duration - sec) * 1e9)))]
        for _ in range(10):
            handle = self.wait_future(self.trajectory.send_goal_async(goal), 5.0, 'trajectory goal')
            if handle.accepted:
                break
            time.sleep(0.3)  # trajectory controller not active yet
        else:
            raise RuntimeError('trajectory goal rejected')
        result = self.wait_future(handle.get_result_async(), duration + 10.0, 'trajectory')
        if result.result.error_code != 0:
            raise RuntimeError(f'trajectory failed: {result.result.error_string}')
        time.sleep(0.5)  # let the image settle

    def wait_for_target(self, prompt, timeout=None):
        end = time.monotonic() + (timeout or self.p['find_timeout'])
        while time.monotonic() < end and not self.cancel.is_set():
            f = self.features
            if f is not None and f.valid and f.prompt == prompt:
                return True
            time.sleep(0.05)
        return False

    def worker(self):
        while rclpy.ok():
            command = self.commands.get()
            self.cancel.clear()
            if not command:
                continue
            try:
                self.execute(command)
            except Exception as exc:  # report and keep serving the next command
                self.report(f'error: {exc}')

    def execute(self, command):
        use_model = self.p['use_model']
        parsed = parse_command(command, self.p['url'] if use_model else None,
                               self.p['model'] if use_model else None)
        if parsed is None:
            self.report(f'not understood: "{command}"')
            return
        action, obj, source = parsed
        if action == STOP:
            self.stop_servoing()
            self.report('stopped')
            return

        self.report(f'"{command}" -> target "{obj}" ({source})')
        self.stop_servoing()
        self.set_prompt(obj)
        self.report(f'searching "{obj}" ...')
        found = self.wait_for_target(obj, self.p['quick_find_timeout'] if self.overview else None)
        if not found and self.overview and not self.cancel.is_set():
            self.report(f'"{obj}" not in view, moving to the overview pose ...')
            self.move_to_overview()
            self.report(f'searching "{obj}" from the overview ...')
            found = self.wait_for_target(obj)
        if not found:
            if not self.cancel.is_set():
                self.report(f'"{obj}" not found')
            return
        answer = self.call(self.start_srv, Trigger.Request())
        self.report(f'moving above "{obj}"' if answer.success else f'start refused: {answer.message}')


def main():
    rclpy.init()
    node = CommandNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


class CommandConsole(Node):
    """Reads commands from the terminal and prints the status of the command node."""

    def __init__(self):
        super().__init__('command_console')
        self.pub = self.create_publisher(String, '/vs/command', 10)
        self.create_subscription(String, '/vs/command_status',
                                 lambda m: print(f'  [{m.data}]', flush=True), 10)


def console():
    rclpy.init()
    node = CommandConsole()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print('Commands, e.g. "fahre über die rote Dose", "move above the blue box", "stop". '
          'Ctrl+D to quit.')
    try:
        while True:
            line = input('> ').strip()
            if line:
                node.pub.publish(String(data=line))
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
