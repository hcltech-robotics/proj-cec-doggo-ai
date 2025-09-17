#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist

from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
from gauge_net_interface.srv import GaugeProcess
from geometry_msgs.msg import Pose, Point, Quaternion

class AprilTagController(Node):
    def __init__(self):
        super().__init__('apriltag_controller')

        # Parameters
        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('desired_distance', 1)  # meters
        self.declare_parameter('desired_y_offset', 0.0)  # meters
        self.declare_parameter('desired_yaw', 0.0)  # radians (0 = directly facing tag)
        self.declare_parameter('position_threshold', 0.25)  # meters
        self.declare_parameter('angle_threshold', 0.25)  # radians
        self.declare_parameter('tag_timeout', 2.0)  # seconds
        self.declare_parameter('optical_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('use_isaac_apriltag', True)

        self.use_isaac_apriltag = self.get_parameter('use_isaac_apriltag').get_parameter_value().bool_value


        # Create publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Set up QoS profile for AprilTag detection subscriber
        tag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        

        # Subscribe to odometry for tracking robot position
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create service
        self.srv = self.create_service(Trigger, 'apriltag_controller/trigger', self.start_control)
        self.navigate = False

        self.gauge_reader = self.create_client(
            GaugeProcess, '/gauge_reader/set_image_process_mode')

        # TF infrastructure
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Control loop timer (10Hz for more responsive control)
        self.timer = self.create_timer(1.0 / 10.0, self.control_loop)

        # State variables initialization
        self.target_tag_id = self.get_parameter('target_tag_id').value
        self.tag_detected = False
        self.tag_position = None
        self.tag_orientation = None
        self.last_detection_time = self.get_clock().now()
        self.position_locked = False

        # TF frame IDs
        self.optical_frame_id = self.get_parameter('optical_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.tag_frame_id = f'apriltag_{self.target_tag_id}'
        self.apriltag_lib_tag_frame_id = f'tag36h11:{self.target_tag_id}'

        # Enhanced TF visualization
        self.publish_tf_visualizations = True

        # Variables for odometry-based tracking
        self.last_odom_pose = None
        self.tag_in_odom_frame = None
        self.tag_orientation_in_odom = None

        # Parameters for position locking
        self.position_threshold = self.get_parameter('position_threshold').value
        self.angle_threshold = self.get_parameter('angle_threshold').value
        self.tag_timeout = self.get_parameter('tag_timeout').value

        if self.use_isaac_apriltag:
            from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
            self.detection_type = AprilTagDetectionArray
            self.detections_topic = '/apriltag/tag_detections'
        else:
            from apriltag_msgs.msg import AprilTagDetectionArray
            self.detection_type = AprilTagDetectionArray
            self.detections_topic = '/apriltag/detections'

        tag_callback = self.tag_callback if self.use_isaac_apriltag else self.tag_callback_lite

        # Subscribe to AprilTag detections from Isaac ROS
        self.tag_sub = self.create_subscription(
            self.detection_type,
            self.detections_topic,
            tag_callback,
            tag_qos
        )

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.nav_goal_handle = None
        self.nav_active = False
        self.desired_distance = self.get_parameter('desired_distance').value
        self.desired_y_offset = self.get_parameter('desired_y_offset').value
        self.desired_yaw = self.get_parameter('desired_yaw').value

        self.target_frame_id = "target"     
        self.current_target_pose = None     


        self.get_logger().info('Simplified AprilTag Controller initialized with odom tracking')

    def start_control(self, request, response):
        """Start the navigation control loop."""

        self.get_logger().info('Starting navigation control loop...')

        self.tag_in_odom_frame = None
        self.tag_orientation_in_odom = None
        self.tag_detected = False
        self.navigate = True
        response.success = True
        response.message = 'Navigation control loop started'
        return response

    def odom_callback(self, msg):
        """Process odometry data."""

        self.last_odom_pose = msg.pose.pose

        # If we have a tag detected and stored in odom frame, update the TF
        if self.tag_in_odom_frame is not None:
            self.publish_tag_transform()
            goal = self.build_approach_goal_pose_in_odom(
                dx=self.desired_distance, dy=self.desired_y_offset, yaw_align=True
            )
            if goal:
                self.current_target_pose = goal
                self.publish_target_transform()

    def publish_target_transform(self):
        """Publish odom -> target for the currently active goal."""
        if self.current_target_pose is None:
            return

        goal = self.current_target_pose
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.target_frame_id

        # position
        t.transform.translation.x = goal.pose.position.x
        t.transform.translation.y = goal.pose.position.y
        t.transform.translation.z = goal.pose.position.z

        # orientation (use the goal’s quaternion)
        t.transform.rotation = goal.pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def publish_tag_transform(self):
        """Publish the transform from odom to tag."""

        if self.tag_in_odom_frame is None or self.tag_orientation_in_odom is None:
            self.get_logger().warn("Skipping publish_tag_transform: tag_in_odom_frame is None")
            return

        # Debug: Print the transform before publishing
        # self.get_logger().info(f"Publishing transform: odom -> {self.tag_frame_id}")
        # self.get_logger().info(f"  Translation: x={self.tag_in_odom_frame[0]}, "
        #                        f"y={self.tag_in_odom_frame[1]}, z={self.tag_in_odom_frame[2]}")
        # self.get_logger().info(f"  Orientation (yaw): {self.tag_orientation_in_odom}")

        # Create transform for odom → tag
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.tag_frame_id

        # Set the translation
        t.transform.translation.x = self.tag_in_odom_frame[0]
        t.transform.translation.y = self.tag_in_odom_frame[1]
        t.transform.translation.z = self.tag_in_odom_frame[2]

        # Set the rotation (from yaw angle)
        cy = math.cos(self.tag_orientation_in_odom * 0.5)
        sy = math.sin(self.tag_orientation_in_odom * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy

        # Publish the transform
        self.tf_broadcaster.sendTransform(t)

        # Also publish a static transform for visualization
        vis_t = TransformStamped()
        vis_t.header.stamp = self.get_clock().now().to_msg()
        vis_t.header.frame_id = self.tag_frame_id
        vis_t.child_frame_id = f'{self.tag_frame_id}_visual'

        # Add a slight offset for better visualization (10cm in front of tag)
        vis_t.transform.translation.x = 0.1  # 10cm in front of tag
        vis_t.transform.translation.y = 0.0
        vis_t.transform.translation.z = 0.0

        # self.get_logger().info(
        # f"VISUAL TRANSFORM - x: {vis_t.transform.translation.x}, y:
        # {vis_t.transform.translation.y}, z: {vis_t.transform.translation.z}")
        vis_t.transform.rotation.x = 0.0
        vis_t.transform.rotation.y = 0.0
        vis_t.transform.rotation.z = 0.0
        vis_t.transform.rotation.w = 1.0

        # Publish the visualization transform
        self.tf_broadcaster.sendTransform(vis_t)

    def publish_enhanced_tf_visualization(self):
        """
        Publish additional TF frames to enhance the AprilTag visualization.
        This is a placeholder in case we want to add more TF frames for better visualization.
        All our main visualization is now handled directly in publish_tag_transform().
        """
        pass

    def build_approach_goal_pose_in_odom(self, dx=0.6, dy=0.0, yaw_align=True):
        """
        Build a goal in odom that is dx meters in front of the TAG (along the line
        from robot -> tag), with optional lateral offset dy (to the robot's left
        if dy>0). Yaw faces the tag, independent of the tag's internal axes.
        """
        if self.tag_in_odom_frame is None or self.last_odom_pose is None:
            return None

        # Positions in odom
        tx, ty, tz = self.tag_in_odom_frame
        rx = self.last_odom_pose.position.x
        ry = self.last_odom_pose.position.y

        # Direction from robot to tag (unit vector)
        yaw_face = math.atan2(ty - ry, tx - rx)   # heading that looks at the tag
        ux = math.cos(yaw_face)
        uy = math.sin(yaw_face)

        # Perpendicular to the left of heading
        px = -uy
        py =  ux

        # Place goal: start at tag, step back by dx along heading toward robot,
        # then slide by dy to the left/right
        gx = tx - dx * ux + dy * px
        gy = ty - dx * uy + dy * py

        # Orientation: face the tag (goal's x+ points toward tag)
        gyaw = yaw_face if yaw_align else 0.0
        cy = math.cos(gyaw * 0.5)
        sy = math.sin(gyaw * 0.5)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.odom_frame_id
        goal.pose.position = Point(x=gx, y=gy, z=0.0)
        goal.pose.orientation = Quaternion(x=0.0, y=0.0, z=sy, w=cy)
        return goal
    
    def send_nav2_goal(self, goal_pose: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("NavigateToPose action server not available")
            return

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        self.get_logger().info(
            f"Sending Nav2 goal in {goal_pose.header.frame_id}: "
            f"x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}"
        )

        self.nav_active = True
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)
    
    def _on_goal_response(self, future):
        self.nav_goal_handle = future.result()
        if not self.nav_goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            self.nav_active = False
            return
        self.get_logger().info("Nav2 goal accepted")
        result_future = self.nav_goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        self.nav_active = False
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f"Nav2 result status: {status}")

        if status == 4:  # SUCCEEDED
            self._finish_and_call_gauge()
        else:
            # Let recoveries happen in BT; you could retry here if desired
            self.get_logger().warn("Nav2 failed; you can implement retry/backoff here")

    def _finish_and_call_gauge(self):
        self.get_logger().info("Target position reached (Nav2).")
        self.navigate = False
        self.call_gauge_read()
    
    def get_pose(self, camera_frame_id):
        #Look up the pose from the tf buffer
        try:
            #self.get_logger().info(f"Looking up transform from {camera_frame_id} to {self.apriltag_lib_tag_frame_id}")
            trans = self.tf_buffer.lookup_transform(
                camera_frame_id,
                self.apriltag_lib_tag_frame_id,
                rclpy.time.Time()
            )

            pose = Pose()
            pose.position = Point(
                x=trans.transform.translation.x,
                y=trans.transform.translation.y,
                z=trans.transform.translation.z
            )
            pose.orientation = Quaternion(
                x=trans.transform.rotation.x,
                y=trans.transform.rotation.y,
                z=trans.transform.rotation.z,
                w=trans.transform.rotation.w
            )
            #self.get_logger().warn(f"Pose: {pose}")
            return pose
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Could not find transform for AprilTag pose: {e}")
            return None

    def tag_callback(self, msg):
        """Process AprilTag detections."""
        for detection in msg.detections:
            if detection.id == self.target_tag_id:
                self.tag_detected = True
                self.last_detection_time = self.get_clock().now()

                # Get tag pose in camera frame
                tag_pose = detection.pose
                camera_frame = tag_pose.header.frame_id

                # Ensure frame_id is not empty; fallback to optical frame
                if not camera_frame:
                    camera_frame = self.optical_frame_id

                try:
                    # Lookup the transform from camera to odom
                    trans = self.tf_buffer.lookup_transform(
                        self.odom_frame_id,  # Target frame (odom)
                        camera_frame,  # Source frame (camera)
                        rclpy.time.Time(),  # Use the latest transform
                        rclpy.duration.Duration(seconds=1.0)  # Allow slight extrapolation
                    )

                    # Transform detected tag position to odom frame
                    tag_pose_odom = do_transform_pose(tag_pose.pose.pose, trans)

                    # Extract position
                    tag_x_odom = tag_pose_odom.position.x
                    tag_y_odom = tag_pose_odom.position.y
                    tag_z_odom = tag_pose_odom.position.z

                    # Store transformed position in odom frame
                    self.tag_in_odom_frame = np.array([tag_x_odom, tag_y_odom, tag_z_odom])

                    # Extract quaternion and convert to yaw
                    q = tag_pose_odom.orientation
                    self.tag_orientation_in_odom = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )

                    # Publish the transform to TF
                    self.publish_tag_transform()

                    self.position_locked = False

                except tf2_ros.LookupException as e:
                    self.get_logger().warn(f"TF Lookup failed: {e}")

                except tf2_ros.ExtrapolationException as e:
                    self.get_logger().warn(f"TF Extrapolation failed: {e}")

                except tf2_ros.ConnectivityException as e:
                    self.get_logger().warn(f"TF Connectivity failed: {e}")


    def control_loop(self):
        if not self.navigate:
            return

        # Tag freshness (reuse your timeout)
        now = self.get_clock().now()
        time_since_detection = (now - self.last_detection_time).nanoseconds / 1e9

        # If we have a tag pose (directly or cached in odom), try to send a goal if none active
        if ((self.tag_detected and time_since_detection < self.tag_timeout) or
            (self.tag_in_odom_frame is not None)):

            if not self.nav_active and self.nav_goal_handle is None:
                goal_pose = self.build_approach_goal_pose_in_odom(
                    dx=self.desired_distance, dy=self.desired_y_offset, yaw_align=True
                )
                if goal_pose:
                    self.send_nav2_goal(goal_pose)
                else:
                    self.get_logger().warn("Could not build approach pose yet; waiting...")
        else:
            if not self.nav_active:
                self.execute_search_behavior()

    def execute_search_behavior(self):
        """Execute search behavior when tag is lost."""
        self.tag_detected = False
        self.position_locked = False

        # Get current time
        now = self.get_clock().now()

        # Rotate for 1 second, then pause for 1.5 seconds to allow for detection
        rotation_period = 2.5  # seconds for a complete rotate-pause cycle
        current_cycle_time = (now.nanoseconds / 1e9) % rotation_period

        if current_cycle_time < 1.0:  # Rotate for 1 second
            # Search behavior - rotate slowly
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.8  # Rotate slowly to search for tag
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('No tag detected, searching... (rotating)')
        else:  # Pause for 1.5 seconds
            # Stop and wait to allow for detection
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('No tag detected, searching... (pausing)')

    def call_gauge_read(self):
        """ Call the gauge reader service asynchronously. """

        # Create request message
        request = GaugeProcess.Request()
        request.process_mode = 1

        self.get_logger().info(f"Sending request: {request}")

        # Call the service asynchronously
        future = self.gauge_reader.call_async(request)
        future.add_done_callback(self.gauge_response_callback)

    def gauge_response_callback(self, future):
        """ Handle the response from the gauge reader service. """
        try:
            response = future.result()
            if response:
                self.get_logger().info(f"Received response: {response}")
            else:
                self.get_logger().error("Failed to receive response.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    controller = AprilTagController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)

        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
