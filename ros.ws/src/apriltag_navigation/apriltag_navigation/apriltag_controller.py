#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from geometry_msgs.msg import Pose, Point, Quaternion
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_geometry_msgs import do_transform_pose
import tf2_ros

from gauge_net_interface.srv import GaugeProcess


class AprilTagController(Node):
    def __init__(self):
        super().__init__('apriltag_controller')

        # ---------- Parameters ----------
        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('desired_distance', 1.0)      # m forward of tag (robot stops here)
        self.declare_parameter('desired_y_offset', 0.0)      # m lateral offset (robot-left positive)
        self.declare_parameter('desired_yaw', 0.0)           # rad (currently unused; we align to tag)
        self.declare_parameter('position_threshold', 0.25)   # m (reserved for future)
        self.declare_parameter('angle_threshold', 0.25)      # rad (reserved for future)
        self.declare_parameter('tag_timeout', 2.0)           # s
        self.declare_parameter('optical_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('use_isaac_apriltag', True)

        # Snapshot parameter values
        self.target_tag_id = self.get_parameter('target_tag_id').value
        self.desired_distance = self.get_parameter('desired_distance').value
        self.desired_y_offset = self.get_parameter('desired_y_offset').value
        self.desired_yaw = self.get_parameter('desired_yaw').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.angle_threshold = self.get_parameter('angle_threshold').value
        self.tag_timeout = self.get_parameter('tag_timeout').value
        self.optical_frame_id = self.get_parameter('optical_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.use_isaac_apriltag = self.get_parameter('use_isaac_apriltag').get_parameter_value().bool_value

        # ---------- Publishers / Subscribers ----------
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        qos_tags = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.tag_sub = self.create_subscription(AprilTagDetectionArray, '/apriltag/tag_detections', self.tag_callback, qos_tags)

        # ---------- Services / Actions ----------
        self.srv = self.create_service(Trigger, 'apriltag_controller/trigger', self.start_control)

        self.gauge_reader = self.create_client(GaugeProcess, '/gauge_reader/set_image_process_mode')
        # Non-blocking: just warn if unavailable (don’t stall node init)
        self.create_timer(1.0, self._warn_if_gauge_unavailable_once)
        self._warned_gauge = False

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.nav_goal_handle = None
        self.nav_active = False
        self.emergency_stop_srv = self.create_service(Trigger, 'apriltag_controller/stop', self.emergency_stop_callback)

        # ---------- TF ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Frames
        self.tag_frame_id = f'apriltag_{self.target_tag_id}'
        self.apriltag_lib_tag_frame_id = f'tag36h11:{self.target_tag_id}'
        self.target_frame_id = 'target'

        # ---------- State ----------
        self.navigate = False
        self.tag_detected = False
        self.last_detection_time = self.get_clock().now()
        self.last_odom_pose: Pose | None = None
        self.tag_in_odom_frame: np.ndarray | None = None   # [x, y, z]
        self.tag_orientation_in_odom: float | None = None  # yaw
        self.current_target_pose: PoseStamped | None = None

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self._last_search_phase = None  # for log de-spam
        self.get_logger().info('AprilTag Controller initialized.')

    # ---------- Triggers ----------
    def start_control(self, _req, resp):
        self.get_logger().info('Starting navigation control loop...')
        self.tag_in_odom_frame = None
        self.tag_orientation_in_odom = None
        self.tag_detected = False
        self.navigate = True
        resp.success = True
        resp.message = 'Navigation control loop started'
        return resp

    def emergency_stop_callback(self, _req, resp):
        self._emergency_stop()
        resp.success = True
        resp.message = 'Emergency stop executed'
        return resp

    # ---------- Odometry ----------
    def odom_callback(self, msg: Odometry):
        self.last_odom_pose = msg.pose.pose

        if self.tag_in_odom_frame is not None:
            self.publish_tag_transform()
            goal = self.build_approach_goal_pose_in_odom(
                dx=self.desired_distance, dy=self.desired_y_offset, yaw_align=True
            )
            if goal:
                self.current_target_pose = goal
                self.publish_target_transform()

    # ---------- TF publishers ----------
    def publish_target_transform(self):
        if self.current_target_pose is None:
            return
        goal = self.current_target_pose

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.target_frame_id
        t.transform.translation.x = goal.pose.position.x
        t.transform.translation.y = goal.pose.position.y
        t.transform.translation.z = goal.pose.position.z
        t.transform.rotation = goal.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def publish_tag_transform(self):
        if self.tag_in_odom_frame is None or self.tag_orientation_in_odom is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.tag_frame_id
        t.transform.translation.x = float(self.tag_in_odom_frame[0])
        t.transform.translation.y = float(self.tag_in_odom_frame[1])
        t.transform.translation.z = float(self.tag_in_odom_frame[2])

        # yaw -> quaternion
        cy = math.cos(self.tag_orientation_in_odom * 0.5)
        sy = math.sin(self.tag_orientation_in_odom * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy

        self.tf_broadcaster.sendTransform(t)

    # ---------- Goal building ----------
    def build_approach_goal_pose_in_odom(self, dx=0.6, dy=0.0, yaw_align=True) -> PoseStamped | None:
        """
        Place a goal 'dx' meters back from the tag (along robot->tag) with lateral offset 'dy'.
        If yaw_align, orient goal to face the tag.
        """
        if self.tag_in_odom_frame is None or self.last_odom_pose is None:
            return None

        tx, ty, _tz = self.tag_in_odom_frame
        rx = self.last_odom_pose.position.x
        ry = self.last_odom_pose.position.y

        yaw_face = math.atan2(ty - ry, tx - rx)  # heading from robot to tag
        ux, uy = math.cos(yaw_face), math.sin(yaw_face)
        px, py = -uy, ux  # left perpendicular

        gx = tx - dx * ux + dy * px
        gy = ty - dx * uy + dy * py

        gyaw = yaw_face if yaw_align else 0.0
        cy, sy = math.cos(gyaw * 0.5), math.sin(gyaw * 0.5)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.odom_frame_id
        goal.pose.position = Point(x=float(gx), y=float(gy), z=0.0)
        goal.pose.orientation = Quaternion(x=0.0, y=0.0, z=sy, w=cy)
        return goal

    # ---------- Nav2 ----------
    def send_nav2_goal(self, goal_pose: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('NavigateToPose action server not available')
            return

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        self.get_logger().info(
            f"Sending Nav2 goal in {goal_pose.header.frame_id}: "
            f"x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}"
        )

        self.nav_active = True
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        self.nav_goal_handle = future.result()
        if not self.nav_goal_handle or not self.nav_goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            self.nav_active = False
            self.nav_goal_handle = None
            return

        self.get_logger().info('Nav2 goal accepted')
        result_future = self.nav_goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        self.nav_active = False
        result = future.result()
        status = result.status if result else -1
        self.get_logger().info(f'Nav2 result status: {status}')
        self.nav_goal_handle = None

        if status == 4:  # SUCCEEDED
            self._finish_and_call_gauge()
        else:
            self.get_logger().warn('Nav2 failed; consider retry/backoff here')

    def _finish_and_call_gauge(self):
        self.get_logger().info('Target position reached (Nav2).')
        self.navigate = False
        self.call_gauge_read()

    def _emergency_stop(self):
        """Immediate stop: cancle Nav2 goal and spam zero cmd_vel."""
        self.get_logger().warn('Emergency stop triggered!')

        # Call emergency stop service if available
        if getattr(self, 'nav_goal_handle', None) and self.nav_goal_handle.accepted:
            try:
                cancel_future = self.nav_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(lambda f: self.get_logger().info('Nav2 goal cancelled'))
            except Exception as e:
                self.get_logger().error(f'Failed to cancel Nav2 goal: {e}')

        self.nav_active = False
        self.navigate = False


        # Spam zero cmd_vel to ensure robot stops
        zero = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Robot stopped.')

    # ---------- Tag callbacks ----------
    def tag_callback(self, msg):
        """Isaac ROS: msg.detections[i].id (int), .pose (PoseWithCovarianceStamped)."""
        for det in msg.detections:
            # Log detection
            if det.id != self.target_tag_id:
                continue
            

            self.tag_detected = True
            self.last_detection_time = self.get_clock().now()

            tag_pose_stamped = det.pose  # PoseWithCovarianceStamped
            camera_frame = tag_pose_stamped.header.frame_id or self.optical_frame_id

            try:
                trans = self.tf_buffer.lookup_transform(
                    self.odom_frame_id, camera_frame, rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0)
                )
                tag_pose_odom = do_transform_pose(tag_pose_stamped.pose.pose, trans)

                self._set_tag_in_odom(tag_pose_odom)
                self.publish_tag_transform()

            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(f'TF error (Isaac): {e}')


    def _set_tag_in_odom(self, pose_in_odom: Pose):
        self.tag_in_odom_frame = np.array([pose_in_odom.position.x,
                                           pose_in_odom.position.y,
                                           pose_in_odom.position.z])
        q = pose_in_odom.orientation
        self.tag_orientation_in_odom = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ---------- Control loop ----------
    def control_loop(self):
        if not self.navigate:
            return

        # freshness
        now = self.get_clock().now()
        time_since_det = (now - self.last_detection_time).nanoseconds / 1e9

        have_tag = (self.tag_detected and time_since_det < self.tag_timeout) or (self.tag_in_odom_frame is not None)

        if have_tag:
            if not self.nav_active and self.nav_goal_handle is None:
                goal_pose = self.build_approach_goal_pose_in_odom(
                    dx=self.desired_distance, dy=self.desired_y_offset, yaw_align=True
                )
                if goal_pose:
                    self.send_nav2_goal(goal_pose)
                else:
                    self.get_logger().warn('Could not build approach pose yet; waiting...')
        else:
            if not self.nav_active:
                self.execute_search_behavior()

    def execute_search_behavior(self):
        """Simple rotate-pause scan when tag is lost."""
        # 2.5s cycle: rotate 1.0s, pause 1.5s
        rotation_period = 2.5
        t = (self.get_clock().now().nanoseconds / 1e9) % rotation_period

        cmd = Twist()
        phase = 'rotate' if t < 1.0 else 'pause'
        if phase == 'rotate':
            cmd.angular.z = 0.8
        # else: zeroed twist (stop)

        self.cmd_vel_pub.publish(cmd)

        # de-spam logs: only print when phase flips
        if phase != self._last_search_phase:
            self.get_logger().info(f'No tag detected, searching... ({phase})')
            self._last_search_phase = phase

    # ---------- Gauge reader ----------
    def call_gauge_read(self):
        req = GaugeProcess.Request()
        req.process_mode = 1
        self.get_logger().info('Requesting gauge read (process_mode=1)')
        future = self.gauge_reader.call_async(req)
        future.add_done_callback(self.gauge_response_callback)

    def gauge_response_callback(self, future):
        try:
            resp = future.result()
            if resp is not None:
                self.get_logger().info(f'Gauge response: {resp}')
            else:
                self.get_logger().error('Gauge response was None.')
        except Exception as e:
            self.get_logger().error(f'Gauge service call failed: {e}')

    def _warn_if_gauge_unavailable_once(self):
        if self._warned_gauge:
            return
        if not self.gauge_reader.service_is_ready():
            self.get_logger().warn('Gauge reader service not available yet.')
        self._warned_gauge = True


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot before shutdown
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
