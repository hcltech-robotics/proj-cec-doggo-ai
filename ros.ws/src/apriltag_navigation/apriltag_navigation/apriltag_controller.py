#!/usr/bin/env python3

import math

from geometry_msgs.msg import Twist
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


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
        # radians (0 = directly facing tag)
        self.declare_parameter('desired_yaw', 0.0)
        self.declare_parameter('position_threshold', 0.25)  # meters
        self.declare_parameter('angle_threshold', 0.25)  # radians

        # Modified PID controllers for lower framerate (6 fps)
        # Increased P for quicker response with fewer updates
        # Decreased D since we have less frequent measurements
        self.linear_pid = PIDController(kp=0.6, ki=0.1, kd=0.1, max_output=0.2)
        self.angular_pid = PIDController(
            kp=1.2, ki=0.1, kd=0.15, max_output=1.0)

        # Create publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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

        # TF listener for tag transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Control loop timer (6Hz to match camera framerate)
        self.timer = self.create_timer(1.0 / 6.0, self.control_loop)

        # State variables
        self.target_tag_id = self.get_parameter('target_tag_id').value
        self.tag_detected = False
        self.tag_position = None
        self.tag_orientation = None
        self.last_detection_time = self.get_clock().now()
        self.position_locked = False

        # Parameters for position locking
        self.position_threshold = self.get_parameter(
            'position_threshold').value
        self.angle_threshold = self.get_parameter('angle_threshold').value

        self.get_logger().info('AprilTag Controller initialized (6 fps mode)')

    def tag_callback(self, msg):
        """Process AprilTag detections."""
        for detection in msg.detections:
            # Check if this is our target tag
            if detection.id == self.target_tag_id:
                self.tag_detected = True
                self.last_detection_time = self.get_clock().now()

                # Get tag pose in camera frame
                tag_pose = detection.pose
                self.tag_position = np.array([
                    tag_pose.pose.pose.position.x,
                    tag_pose.pose.pose.position.y,
                    tag_pose.pose.pose.position.z
                ])

                # Convert quaternion to yaw angle (around Z axis)
                q = tag_pose.pose.pose.orientation
                self.tag_orientation = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )

                msg = f'Tag detected at position: {self.tag_position}, '
                msg += f'orientation: {self.tag_orientation}'
                self.get_logger().info(msg)

                # Reset locked state when we get a new detection
                self.position_locked = False

    def control_loop(self):
        """Loop main control loop for navigating toward the tag."""
        # Check if we've seen the tag recently (within 2 seconds - longer for
        # lower framerate)
        now = self.get_clock().now()
        time_since_detection = (
            now - self.last_detection_time).nanoseconds / 1e9

        if self.tag_detected and time_since_detection < 2.0:
            # Calculate distance to tag
            distance = np.linalg.norm(self.tag_position)

            # Distance error (positive means we need to move forward)
            distance_error = distance - \
                self.get_parameter('desired_distance').value

            # Angle to tag (positive means tag is to the left)
            angle_to_tag = math.atan2(
                self.tag_position[0], self.tag_position[2])

            # Check if we've reached the target position (for drift correction)
            if (abs(distance_error) < self.position_threshold and
                    abs(angle_to_tag) < self.angle_threshold):

                if not self.position_locked:
                    self.position_locked = True
                    self.get_logger().info('Position locked, drift corrected')

                # Stop the robot once position is achieved
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                return

            # Position not locked yet, continue with PID control
            # Compute PID control outputs with longer dt due to lower framerate
            linear_vel = self.linear_pid.compute(distance_error, 1.0 / 6.0)
            angular_vel = -self.angular_pid.compute(angle_to_tag, 1.0 / 6.0)

            # Create and publish velocity command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)

            debug_msg = f'Distance: {distance:.2f}m, Angle: {angle_to_tag:.2f}rad, '
            debug_msg += f'Commands: linear={linear_vel:.2f}, angular={angular_vel:.2f}'
            self.get_logger().info(debug_msg)
        else:
            # If we haven't seen the tag recently, stop or search
            self.tag_detected = False
            self.position_locked = False

            # Simple search behavior: rotate slowly to find the tag
            cmd = Twist()
            cmd.linear.x = 0.0
            # cmd.angular.z = 0.3  # Rotate slowly to search for tag
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)

            self.get_logger().info(
                f'No tag detected searching... td={self.tag_detected} tsd={time_since_detection:.2f}s')


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
