#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
from gauge_net_interface.srv import GaugeProcess
from geometry_msgs.msg import Pose, Point, Quaternion

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.previous_error = 0.0
        self.integral = 0.0
        self.integral_limit = 1.0

    def compute(self, error, dt):
        """Compute PID control output."""
        # P term
        p_term = self.kp * error

        # I term (with anti-windup)
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral

        # D term
        d_term = 0.0
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error

        # Calculate total control output with limiting
        output = p_term + i_term + d_term
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
        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('desired_distance', 0.7)  # meters
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

        # PID controllers for 3-DOF control
        self.linear_x_pid = PIDController(kp=0.6, ki=0.1, kd=0.1, max_output=0.2)
        self.linear_y_pid = PIDController(kp=0.6, ki=0.1, kd=0.1, max_output=0.2)
        self.angular_pid = PIDController(kp=1.2, ki=0.1, kd=0.15, max_output=1.0)

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
    
    def tag_callback_lite(self, msg):
        """Process AprilTag detections."""
        for detection in msg.detections:
            if detection.id == self.target_tag_id:
                #self.get_logger().warn(f"Received apriltag detection: {detection}")
                self.tag_detected = True
                self.last_detection_time = self.get_clock().now()

                camera_frame = msg.header.frame_id
                # Ensure frame_id is not empty; fallback to optical frame
                if not camera_frame:
                    camera_frame = self.optical_frame_id
               
                # Get tag pose in camera frame
                tag_pose = self.get_pose(camera_frame)
                if tag_pose is None:
                    return

                try:
                    # Lookup the transform from camera to odom
                    trans = self.tf_buffer.lookup_transform(
                        self.odom_frame_id,  # Target frame (odom)
                        camera_frame,  # Source frame (camera)
                        rclpy.time.Time(),  # Use the latest transform
                        rclpy.duration.Duration(seconds=1.0)  # Allow slight extrapolation
                    )

                    # Transform detected tag position to odom frame
                    tag_pose_odom = do_transform_pose(tag_pose, trans)

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

    def get_relative_tag_pose(self):
        """Get the tag pose relative to the robot's current position."""
        if self.tag_in_odom_frame is not None and self.last_odom_pose is not None:
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

                rel_position = np.array([
                    x_local,   # Forward (X in robot frame)
                    y_local,   # Left/right (Y in robot frame)
                    z_local    # Up/down (Z in robot frame)
                ])

                # Calculate relative orientation difference
                rel_orientation = self.tag_orientation_in_odom - robot_yaw

                # Apply -π/2 correction
                rel_orientation += np.pi / 2  # Shift by 90 degrees

                # Normalize to [-pi, pi]
                rel_orientation = (rel_orientation + np.pi) % (2 * np.pi) - np.pi

                return rel_position, self.tag_orientation_in_odom, rel_orientation
            except Exception as e:
                self.get_logger().error(f'Error getting relative pose from odom: {e}')
                return None, None, None

        return None, None, None

    def control_loop(self):
        """Main control loop for navigating toward the tag."""
        if not self.navigate:
            return

        # Check if we've seen the tag recently
        now = self.get_clock().now()
        time_since_detection = (now - self.last_detection_time).nanoseconds / 1e9

        if (self.tag_detected and time_since_detection <
                self.tag_timeout) or self.tag_in_odom_frame is not None:
            # Get tag position relative to robot
            rel_position, rel_orientation, angle_to_tag = self.get_relative_tag_pose()

            if rel_position is None:
                self.get_logger().warning('Could not determine relative tag position')
                self.execute_search_behavior()
                return

            # X distance error (forward/backward)
            x_distance_error = rel_position[0] - self.get_parameter('desired_distance').value

            # Y offset error (left/right)
            y_distance_error = rel_position[1] - self.get_parameter('desired_y_offset').value

            # Angle error (orientation difference between robot and tag)
            yaw_error = angle_to_tag - self.get_parameter('desired_yaw').value

            # Normalize yaw_error to [-pi, pi]
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            # Check if we've reached the target position
            if (abs(x_distance_error) < self.position_threshold and
                    abs(y_distance_error) < self.position_threshold and
                    abs(yaw_error) < self.angle_threshold):

                if not self.position_locked:
                    self.position_locked = True
                    self.get_logger().info('Position locked!')

                # Stop the robot once position is achieved
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.linear.y = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)

                self.navigate = False
                self.get_logger().info('Target position reached!')

                self.call_gauge_read()
                return

            # Position not locked yet, continue with PID control for 3-DOF
            linear_x_vel = self.linear_x_pid.compute(x_distance_error, 1.0 / 10.0)
            linear_y_vel = self.linear_y_pid.compute(y_distance_error, 1.0 / 10.0)
            angular_vel = self.angular_pid.compute(yaw_error, 1.0 / 10.0) / 10.0

            # Create and publish velocity command
            cmd = Twist()
            cmd.linear.x = linear_x_vel
            cmd.linear.y = linear_y_vel

            if abs(yaw_error) < self.angle_threshold:
                cmd.angular.z = 0.0
            elif abs(x_distance_error) < 1.0:
                cmd.angular.z = angular_vel

            self.cmd_vel_pub.publish(cmd)

            debug_msg = f'Distance X: {x_distance_error:.2f}m, Y: {y_distance_error:.2f}m, '
            debug_msg += f'Angle: {yaw_error:.2f}rad, Commands: x={linear_x_vel:.2f}, '
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

        # Rotate for 1 second, then pause for 1.5 seconds to allow for detection
        rotation_period = 2.5  # seconds for a complete rotate-pause cycle
        current_cycle_time = (now.nanoseconds / 1e9) % rotation_period

        if current_cycle_time < 1.0:  # Rotate for 1 second
            # Search behavior - rotate slowly
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.8  # Rotate slowly to search for tag
            #self.cmd_vel_pub.publish(cmd)
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
