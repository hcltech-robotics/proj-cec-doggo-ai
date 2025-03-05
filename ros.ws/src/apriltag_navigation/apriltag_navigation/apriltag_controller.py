#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker


# flake8: noqa: E501

class PIDController:

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self.previous_error = 0.0
        self.integral = 0.0

        # Add anti-windup limit
        self.integral_limit = 1.0

    def compute(self, error, dt):
        """Compute PID control output."""
        # P term
        p_term = self.kp * error

        # I term (with anti-windup)
        self.integral += error * dt

        # Apply integral limiting to prevent windup
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        i_term = self.ki * self.integral

        # D term
        d_term = 0.0
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt

        # Save state for next iteration
        self.previous_error = error

        # Calculate total control output with limiting
        output = p_term + i_term + d_term

        # Apply output limiting
        if output > self.max_output:
            output = self.max_output
        elif output < -self.max_output:
            output = -self.max_output

        return output

    def reset(self):
        """Reset the controller state."""
        self.previous_error = 0.0
        self.integral = 0.0


class AprilTagController(Node):

    def __init__(self):
        super().__init__('apriltag_controller')

        # Parameters
        self.declare_parameter('target_tag_id', 1)
        self.declare_parameter('desired_distance', 0.5)  # meters
        self.declare_parameter('desired_y_offset', 0.0)  # meters
        # radians (0 = directly facing tag)
        self.declare_parameter('desired_yaw', 0.0)
        self.declare_parameter('position_threshold', 0.25)  # meters
        self.declare_parameter('angle_threshold', 0.25)  # radians
        self.declare_parameter('tag_timeout', 2.0)  # seconds
        self.declare_parameter('use_odom', True)  # Use odometry for tracking
        self.declare_parameter(
            'optical_frame_id',
            'camera_color_optical_frame')  # Camera optical frame ID
        self.declare_parameter(
            'base_frame_id',
            'base_link')  # Robot base frame ID
        self.declare_parameter('odom_frame_id', 'odom')  # Odometry frame ID

        # PID controllers for 3-DOF control
        self.linear_x_pid = PIDController(
            kp=0.6, ki=0.1, kd=0.1, max_output=0.2)
        self.linear_y_pid = PIDController(
            kp=0.6, ki=0.1, kd=0.1, max_output=0.2)
        self.angular_pid = PIDController(
            kp=1.2, ki=0.1, kd=0.15, max_output=1.0)

        # Create publisher for robot velocity commands and visualization
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/apriltag_marker', 10)

        # Set up QoS profile for AprilTag detection subscriber
        tag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to AprilTag detections from Isaac ROS
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/apriltag/tag_detections',
            self.tag_callback,
            tag_qos
        )

        # Subscribe to odometry for tracking robot position
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # TF infrastructure
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Control loop timer (6Hz to match camera framerate)
        self.timer = self.create_timer(1.0 / 6.0, self.control_loop)

        # State variables initialization
        self.target_tag_id = self.get_parameter('target_tag_id').value
        self.tag_detected = False
        self.tag_position = None
        self.tag_orientation = None
        self.last_detection_time = self.get_clock().now()
        self.position_locked = False
        self.use_odom = self.get_parameter('use_odom').value

        # TF frame IDs
        self.optical_frame_id = self.get_parameter('optical_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.tag_frame_id = f'apriltag_{self.target_tag_id}'

        # Additional state variables for visualization
        self.publish_marker = True  # Whether to publish marker for visualization

        # Variables for odometry-based tracking
        self.last_odom_pose = None
        self.tag_in_odom_frame = None
        self.tag_orientation_in_odom = None

        # Parameters for position locking
        self.position_threshold = self.get_parameter(
            'position_threshold').value
        self.angle_threshold = self.get_parameter('angle_threshold').value
        self.tag_timeout = self.get_parameter('tag_timeout').value

        # Wait a moment for TF tree to become available
        self.get_logger().info('Waiting for TF tree to be populated...')
        self.tf_check_timer = self.create_timer(
            1.0, self.check_tf_tree_and_cancel_timer)

        self.get_logger().info(
            'Improved AprilTag Controller initialized with y-axis control and odometry tracking')

    def check_tf_tree_and_cancel_timer(self):
        """Check TF tree and then cancel the timer to make it a one-shot."""
        self.check_tf_tree()
        # Cancel the timer to make it run only once
        self.tf_check_timer.cancel()

    def check_tf_tree(self):
        """Check if necessary frames exist in the TF tree."""
        try:
            # List all available frames from TF
            frames = self.tf_buffer.all_frames_as_string()
            self.get_logger().info(f'Available TF frames:\n{frames}')

            # Try to get transform from base frame to optical frame
            try:
                if self.tf_buffer.can_transform(
                    self.base_frame_id,
                    self.optical_frame_id,
                    self.get_clock().now(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                ):
                    self.get_logger().info(
                        f'Successfully found transform from {self.optical_frame_id} to {self.base_frame_id}')

                    # Get and print the transform for debugging
                    transform = self.tf_buffer.lookup_transform(
                        self.base_frame_id,
                        self.optical_frame_id,
                        self.get_clock().now(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    self.get_logger().info(
                        f'Transform: {transform.transform.translation.x}, '
                        f'{transform.transform.translation.y}, {transform.transform.translation.z}')
                else:
                    self.get_logger().warn(
                        f'No transform from {self.optical_frame_id} to {self.base_frame_id}')
            except Exception as e:
                self.get_logger().warn(
                    f'Error checking optical to base transform: {e}')

            # Try to get transform from base frame to odom
            try:
                if self.tf_buffer.can_transform(
                    self.odom_frame_id,
                    self.base_frame_id,
                    self.get_clock().now(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                ):
                    self.get_logger().info(
                        f'Successfully found transform from {self.base_frame_id} to {self.odom_frame_id}')
                else:
                    self.get_logger().warn(
                        f'No transform from {self.base_frame_id} to {self.odom_frame_id}')
            except Exception as e:
                self.get_logger().warn(
                    f'Error checking base to odom transform: {e}')

            # Try direct transform from optical to odom
            try:
                if self.tf_buffer.can_transform(
                    self.odom_frame_id,
                    self.optical_frame_id,
                    self.get_clock().now(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                ):
                    self.get_logger().info(
                        f'Successfully found direct transform from {self.optical_frame_id} to {self.odom_frame_id}')
                else:
                    self.get_logger().warn(
                        f'No direct transform from {self.optical_frame_id} to {self.odom_frame_id}')
            except Exception as e:
                self.get_logger().warn(
                    f'Error checking optical to odom transform: {e}')

        except Exception as e:
            self.get_logger().error(f'Error checking TF tree: {e}')

    def odom_callback(self, msg):
        """Process odometry data."""
        self.last_odom_pose = msg.pose.pose

        # If we have a tag detected and stored in odom frame, update the TF
        if self.tag_in_odom_frame is not None:
            self.publish_tag_transform()

    def publish_tag_transform(self):
        """Publish the transform from odom to tag."""
        if self.tag_in_odom_frame is None or self.tag_orientation_in_odom is None:
            return

        # Create transform for odom → tag
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.tag_frame_id

        # Set the translation
        t.transform.translation.x = self.tag_in_odom_frame[0]
        t.transform.translation.y = self.tag_in_odom_frame[1]
        t.transform.translation.z = self.tag_in_odom_frame[2]

        # Set the rotation (directly from yaw angle)
        # Convert yaw to quaternion manually
        cy = math.cos(self.tag_orientation_in_odom * 0.5)
        sy = math.sin(self.tag_orientation_in_odom * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy

        # Publish the transform
        self.tf_broadcaster.sendTransform(t)

        # Also publish a static transform for visualization with a fixed offset
        # This makes the tag more visible in RViz/Foxglove
        vis_t = TransformStamped()
        vis_t.header.stamp = self.get_clock().now().to_msg()
        vis_t.header.frame_id = self.tag_frame_id
        vis_t.child_frame_id = f'{self.tag_frame_id}_visual'

        # Small offset in the tag's forward direction
        vis_t.transform.translation.x = 0.0
        vis_t.transform.translation.y = 0.0
        vis_t.transform.translation.z = 0.0

        # Identity rotation
        vis_t.transform.rotation.x = 0.0
        vis_t.transform.rotation.y = 0.0
        vis_t.transform.rotation.z = 0.0
        vis_t.transform.rotation.w = 1.0

        # Publish the visualization transform
        self.tf_broadcaster.sendTransform(vis_t)

    def publish_visualization_marker(self):
        """Publish a visualization marker for the AprilTag in RViz/Foxglove."""
        if not self.publish_marker or self.tag_in_odom_frame is None:
            return

        marker = Marker()
        marker.header.frame_id = self.odom_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'apriltag'
        marker.id = self.target_tag_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set marker position to tag position
        marker.pose.position.x = self.tag_in_odom_frame[0]
        marker.pose.position.y = self.tag_in_odom_frame[1]
        marker.pose.position.z = self.tag_in_odom_frame[2]

        # Convert yaw to quaternion for the marker
        cy = math.cos(self.tag_orientation_in_odom * 0.5)
        sy = math.sin(self.tag_orientation_in_odom * 0.5)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = sy
        marker.pose.orientation.w = cy

        # Set marker scale - make it a visible tag-sized cube
        marker.scale.x = 0.15  # Tag width (typically ~15cm)
        marker.scale.y = 0.15  # Tag height
        marker.scale.z = 0.01  # Tag thickness

        # Set marker color - bright green for visibility
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Slightly transparent

        # Set lifetime for the marker
        marker.lifetime.sec = 1  # 1 second lifetime

        # Publish the marker
        self.marker_pub.publish(marker)

    def tag_callback(self, msg):
        """Process AprilTag detections."""
        for detection in msg.detections:
            # Check if this is our target tag
            if detection.id == self.target_tag_id:
                self.tag_detected = True
                self.last_detection_time = self.get_clock().now()

                # Get tag pose in camera frame
                tag_pose = detection.pose

                # Debug detected position
                self.get_logger().debug(
                    f'Raw tag position: [{tag_pose.pose.pose.position.x}, {tag_pose.pose.pose.position.y}, {tag_pose.pose.pose.position.z}]')

                # In optical frame, Z is forward, X is right, Y is down
                # Convert to robot frame where X is forward, Y is left
                self.tag_position = np.array([
                    tag_pose.pose.pose.position.z,
                    # Forward (Z in optical frame -> X in robot frame)
                    -tag_pose.pose.pose.position.x,
                    # Left/right (negative X in optical frame -> Y in robot
                    # frame)
                    # Up/down (negative Y in optical frame -> Z in robot frame)
                    -tag_pose.pose.pose.position.y
                ])

                # Convert quaternion to yaw angle (around Z axis)
                # Need to handle optical frame to robot frame conversion
                q = tag_pose.pose.pose.orientation
                # Extract rotation from quaternion (simplified for optical
                # frame)
                self.tag_orientation = math.atan2(
                    2.0 * (q.w * q.y - q.x * q.z),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )

                # If using odometry, store the tag position relative to odom
                # frame
                if self.use_odom and self.last_odom_pose is not None:
                    try:
                        # Get camera frame ID from the message header
                        camera_frame = tag_pose.header.frame_id
                        if not camera_frame:
                            # If frame_id is empty, use the default optical
                            # frame
                            camera_frame = self.optical_frame_id
                            self.get_logger().warn(
                                f'Empty frame_id in tag detection. Using default: {camera_frame}')

                        # Log the actual frame being used
                        self.get_logger().info(
                            f'Tag detected in frame: {camera_frame}')

                        # Get the current time for transformations
                        now = self.get_clock().now()

                        # Create a pose stamped with the tag position in camera
                        # frame
                        tag_pose_stamped = PoseStamped()
                        tag_pose_stamped.header = tag_pose.header
                        tag_pose_stamped.pose = tag_pose.pose.pose

                        # If the TF tree doesn't have the needed transforms, we'll construct
                        # the transform manually using the tag detection and
                        # odometry

                        # First, check if we can use the TF tree (preferred
                        # method)
                        try_tf_tree = False

                        if try_tf_tree:
                            try:
                                if self.tf_buffer.can_transform(
                                    self.odom_frame_id,
                                    camera_frame,
                                    now,
                                    timeout=rclpy.duration.Duration(
                                        seconds=0.1)):
                                    tag_in_odom = self.tf_buffer.transform(
                                        tag_pose_stamped, self.odom_frame_id)

                                    # Store the tag position in odom frame
                                    self.tag_in_odom_frame = np.array([
                                        tag_in_odom.pose.position.x,
                                        tag_in_odom.pose.position.y,
                                        tag_in_odom.pose.position.z
                                    ])

                                    # Extract orientation in odom frame
                                    q_odom = tag_in_odom.pose.orientation
                                    self.tag_orientation_in_odom = math.atan2(
                                        2.0 * (q_odom.w * q_odom.z + q_odom.x * q_odom.y),
                                        1.0 - 2.0 * (q_odom.y * q_odom.y + q_odom.z * q_odom.z)
                                    )

                                    # Publish the tag transform to TF
                                    self.publish_tag_transform()

                                    self.get_logger().debug(
                                        f'Tag stored in odom frame at: {self.tag_in_odom_frame} (direct transform)')
                                    return  # Successfully transformed
                            except Exception as e:
                                self.get_logger().warning(
                                    f'Direct transform failed: {e}')

                        # Manual transform from tag detection to odom frame (using robot's current position)
                        # This works when the TF tree doesn't provide the
                        # needed transforms
                        try:
                            if self.last_odom_pose is not None:
                                # Extract robot position from odom
                                robot_pos = np.array([
                                    self.last_odom_pose.position.x,
                                    self.last_odom_pose.position.y,
                                    self.last_odom_pose.position.z
                                ])

                                # Extract robot orientation (quaternion)
                                robot_quat = [
                                    self.last_odom_pose.orientation.x,
                                    self.last_odom_pose.orientation.y,
                                    self.last_odom_pose.orientation.z,
                                    self.last_odom_pose.orientation.w
                                ]

                                # Calculate yaw from quaternion
                                robot_yaw = math.atan2(
                                    2.0 *
                                    (robot_quat[3] * robot_quat[2] + robot_quat[0] * robot_quat[1]),
                                    1.0 - 2.0 * (robot_quat[1] ** 2 + robot_quat[2] ** 2))

                                # Calculate rotation matrix from yaw
                                cos_yaw = math.cos(robot_yaw)
                                sin_yaw = math.sin(robot_yaw)

                                # Apply rotation to tag position (in robot
                                # frame) to get odom frame position
                                # Already converted from optical to robot frame
                                tag_in_robot_frame = self.tag_position

                                # Rotate and translate to odom frame
                                tag_x_odom = robot_pos[0] + tag_in_robot_frame[0] * \
                                    cos_yaw - tag_in_robot_frame[1] * sin_yaw
                                tag_y_odom = robot_pos[1] + tag_in_robot_frame[0] * \
                                    sin_yaw + tag_in_robot_frame[1] * cos_yaw
                                tag_z_odom = robot_pos[2] + \
                                    tag_in_robot_frame[2]

                                # Store tag position in odom frame
                                self.tag_in_odom_frame = np.array(
                                    [tag_x_odom, tag_y_odom, tag_z_odom])

                                # Calculate tag orientation in odom frame
                                self.tag_orientation_in_odom = robot_yaw + self.tag_orientation

                                # Normalize angle to [-pi, pi]
                                while self.tag_orientation_in_odom > math.pi:
                                    self.tag_orientation_in_odom -= 2 * math.pi
                                while self.tag_orientation_in_odom < -math.pi:
                                    self.tag_orientation_in_odom += 2 * math.pi

                                # Publish the transform
                                self.publish_tag_transform()

                                self.get_logger().info(
                                    f'Tag stored in odom frame at: {self.tag_in_odom_frame} (manual transform)')
                                return  # Successfully transformed
                        except Exception as e:
                            self.get_logger().warning(
                                f'Manual transform failed: {e}')

                        # Fallback: Try two-step transformation
                        # (optical→base→odom)
                        try:
                            # First to base_link
                            if self.tf_buffer.can_transform(
                                self.base_frame_id,
                                camera_frame,
                                now,
                                timeout=rclpy.duration.Duration(
                                    seconds=0.1)):
                                tag_in_base = self.tf_buffer.transform(
                                    tag_pose_stamped, self.base_frame_id)

                                # Then to odom
                                if self.tf_buffer.can_transform(
                                    self.odom_frame_id,
                                    self.base_frame_id,
                                    now,
                                    timeout=rclpy.duration.Duration(
                                        seconds=0.1)):
                                    tag_in_odom = self.tf_buffer.transform(
                                        tag_in_base, self.odom_frame_id)

                                    # Store the tag position in odom frame
                                    self.tag_in_odom_frame = np.array([
                                        tag_in_odom.pose.position.x,
                                        tag_in_odom.pose.position.y,
                                        tag_in_odom.pose.position.z
                                    ])

                                    # Extract orientation in odom frame
                                    q_odom = tag_in_odom.pose.orientation
                                    self.tag_orientation_in_odom = math.atan2(
                                        2.0 * (q_odom.w * q_odom.z + q_odom.x * q_odom.y),
                                        1.0 - 2.0 * (q_odom.y * q_odom.y + q_odom.z * q_odom.z)
                                    )

                                    # Publish the tag transform to TF
                                    self.publish_tag_transform()

                                    self.get_logger().debug(
                                        f'Tag stored in odom frame at: {self.tag_in_odom_frame} (two-step transform)')
                                    return  # Successfully transformed
                                else:
                                    self.get_logger().warning(
                                        f'Cannot transform from {self.base_frame_id} to {self.odom_frame_id}')
                            else:
                                self.get_logger().warning(
                                    f'Cannot transform from {camera_frame} to {self.base_frame_id}')
                        except Exception as e:
                            self.get_logger().warning(
                                f'Two-step transform failed: {e}')

                    except Exception as e:
                        self.get_logger().warning(
                            f'Failed to transform tag to odom: {e}')

                msg = f'Tag detected at position: {self.tag_position}, '
                msg += f'orientation: {self.tag_orientation}'
                self.get_logger().info(msg)

                # Also publish a visualization marker for RViz/Foxglove
                self.publish_visualization_marker()

                # Reset locked state when we get a new detection
                self.position_locked = False

    def get_relative_tag_pose(self):
        """Get the tag pose relative to the robot's current position."""
        if self.use_odom and self.tag_in_odom_frame is not None and self.last_odom_pose is not None:
            try:
                # Get transform from tag to robot in odom frame
                robot_position = np.array([
                    self.last_odom_pose.position.x,
                    self.last_odom_pose.position.y,
                    self.last_odom_pose.position.z
                ])

                # Extract robot orientation from quaternion
                q_robot = self.last_odom_pose.orientation
                robot_yaw = math.atan2(
                    2.0 * (q_robot.w * q_robot.z + q_robot.x * q_robot.y),
                    1.0 - 2.0 * (q_robot.y * q_robot.y + q_robot.z * q_robot.z)
                )

                # Calculate relative position of tag from robot in odom frame
                dx = self.tag_in_odom_frame[0] - robot_position[0]
                dy = self.tag_in_odom_frame[1] - robot_position[1]

                # Rotate the position to robot's local frame
                cos_yaw = math.cos(-robot_yaw)
                sin_yaw = math.sin(-robot_yaw)

                x_local = dx * cos_yaw - dy * sin_yaw
                y_local = dx * sin_yaw + dy * cos_yaw
                z_local = self.tag_in_odom_frame[2] - robot_position[2]

                # For the odom-based approach, ensure we're properly transforming coordinates
                # Here we're converting from robot base frame to odom frame
                rel_position = np.array([
                    x_local,   # Forward (X in robot frame)
                    y_local,   # Left/right (Y in robot frame)
                    z_local    # Up/down (Z in robot frame)
                ])

                # Calculate relative angle
                angle_to_tag = math.atan2(rel_position[1], rel_position[0])

                return rel_position, self.tag_orientation_in_odom, angle_to_tag
            except Exception as e:
                self.get_logger().error(
                    f'Error getting relative pose from odom: {e}')
                return None, None, None

        # Fallback to direct camera-based detection
        return self.tag_position, self.tag_orientation, math.atan2(
            self.tag_position[1], self.tag_position[0]) if self.tag_position is not None else None

    def control_loop(self):
        """Loop main control loop for navigating toward the tag."""
        # Check if we've seen the tag recently
        now = self.get_clock().now()
        time_since_detection = (
            now - self.last_detection_time).nanoseconds / 1e9

        if (self.tag_detected and time_since_detection < self.tag_timeout) or \
           (self.use_odom and self.tag_in_odom_frame is not None):

            # Get tag position relative to robot (either from direct detection
            # or odom-based tracking)
            rel_position, rel_orientation, angle_to_tag = self.get_relative_tag_pose()

            if rel_position is None:
                self.get_logger().warning('Could not determine relative tag position')
                self.execute_search_behavior()
                return

            # X distance error (forward/backward)
            x_distance_error = rel_position[0] - \
                self.get_parameter('desired_distance').value

            # Y offset error (left/right)
            y_distance_error = rel_position[1] - \
                self.get_parameter('desired_y_offset').value

            # Debug output to verify angle calculation
            self.get_logger().debug(
                f'Angle calculation: atan2({rel_position[1]}, {rel_position[0]}) = {angle_to_tag}')

            # Check if we've reached the target position (for drift correction)
            if (abs(x_distance_error) < self.position_threshold and
                    abs(y_distance_error) < self.position_threshold and
                    abs(angle_to_tag) < self.angle_threshold):

                if not self.position_locked:
                    self.position_locked = True
                    self.get_logger().info('Position locked!')

                # Stop the robot once position is achieved
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.linear.y = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                return

            # Position not locked yet, continue with PID control for 3-DOF
            linear_x_vel = self.linear_x_pid.compute(
                x_distance_error, 1.0 / 6.0)
            linear_y_vel = self.linear_y_pid.compute(
                y_distance_error, 1.0 / 6.0)
            angular_vel = self.angular_pid.compute(angle_to_tag, 1.0 / 6.0)

            # Create and publish velocity command
            cmd = Twist()
            cmd.linear.x = linear_x_vel
            cmd.linear.y = linear_y_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)

            debug_msg = f'Distance X: {x_distance_error:.2f}m, Y: {y_distance_error:.2f}m, '
            debug_msg += f'Angle: {angle_to_tag:.2f}rad, Commands: x={linear_x_vel:.2f}, '
            debug_msg += f'y={linear_y_vel:.2f}, angular={angular_vel:.2f}'
            self.get_logger().info(debug_msg)
        else:
            self.execute_search_behavior()

    def execute_search_behavior(self):
        """Execute search behavior when tag is lost."""
        self.tag_detected = False
        self.position_locked = False

        # Get current time
        now = self.get_clock().now()

        # Create a more sophisticated search pattern with pauses
        # Rotate for 1 second, then pause for 1.5 seconds to allow for
        # detection
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
        else:  # Pause for 0.5 seconds
            # Stop and wait to allow for detection
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info('No tag detected, searching... (pausing)')


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
