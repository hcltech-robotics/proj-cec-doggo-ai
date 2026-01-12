#!/usr/bin/env python3

import base64

import cv2
import requests

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.qos_overriding_options import QoSProfile
from gauge_net_interface.msg import GaugeReading
from gauge_net_interface.srv import GaugeProcess
from sensor_msgs.msg import Image

# from gauge_net.transforms import custom_transform


from .gauge_reader_parent import GaugeReaderParent


class GaugeReaderNode(GaugeReaderParent):
    def __init__(self):
        self._namespace = 'gauge_reader'
        self._node_name = 'gauge_reader'
        Node.__init__(self, self._node_name, namespace=self._namespace)

        # Use the GPU if available.
        # self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare and get node parameters.
        self.declare_parameters(
            namespace=self._namespace,
            parameters=[
                ('use_math', True),
                ('image_topic', '/apriltag/image_rect'),
                ('model_server_url', ''),
                ('token', ''),
                ('min_gauge_score', 0.99),
                ('min_needle_score', 0.95),
                ('scaling_min', 0.0),
                ('scaling_max', 10.0),
                ('image_stream.reliability', rclpy.Parameter.Type.STRING),
                ('image_stream.history', rclpy.Parameter.Type.STRING),
                ('image_stream.depth', rclpy.Parameter.Type.INTEGER),
            ],
        )

        self._use_math_reading = (
            self.get_parameter(f'{self._namespace}.use_math')
            .get_parameter_value()
            .bool_value
        )
        self._image_topic = (
            self.get_parameter(f'{self._namespace}.image_topic')
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(f'Image Topic: {self._image_topic}')
        self._min_gauge_score = (
            self.get_parameter(f'{self._namespace}.min_gauge_score')
            .get_parameter_value()
            .double_value
        )
        self._min_needle_score = (
            self.get_parameter(f'{self._namespace}.min_needle_score')
            .get_parameter_value()
            .double_value
        )
        self._scaling_min = (
            self.get_parameter(f'{self._namespace}.scaling_min')
            .get_parameter_value()
            .double_value
        )
        self._scaling_max = (
            self.get_parameter(f'{self._namespace}.scaling_max')
            .get_parameter_value()
            .double_value
        )
        self.model_server_url = (
            self.get_parameter(f'{self._namespace}.model_server_url')
            .get_parameter_value()
            .string_value
        )
        self.token = (
            self.get_parameter(f'{self._namespace}.token')
            .get_parameter_value()
            .string_value
        )
        image_reliability = self.get_parameter(
            f'{self._namespace}.image_stream.reliability'
        ).value
        image_history = self.get_parameter(
            f'{self._namespace}.image_stream.history'
        ).value
        image_depth = self.get_parameter(f'{self._namespace}.image_stream.depth').value
        self.get_logger().info(str(self._use_math_reading))

        # Setting up QoS profile for the image subscriber.
        RELIABILITY_MAP = {
            'best_effort': ReliabilityPolicy.BEST_EFFORT,
            'reliable': ReliabilityPolicy.RELIABLE,
        }

        HISTORY_MAP = {
            'keep_all': HistoryPolicy.KEEP_ALL,
            'keep_last': HistoryPolicy.KEEP_LAST,
        }
        self.get_logger().info(
            f'Image QoS settings - Reliability: {image_reliability}, '
            + 'History: {image_history}, Depth: {image_depth}'
        )

        image_qos_profile = QoSProfile(
            reliability=RELIABILITY_MAP.get(
                image_reliability, ReliabilityPolicy.RELIABLE
            ),
            history=HISTORY_MAP.get(image_history, HistoryPolicy.KEEP_LAST),
            durability=DurabilityPolicy.VOLATILE,
            depth=image_depth,
        )

        # self._detector_transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )

        # # Image processing pipeline for the reader.
        # self._reader_transform = transforms.Compose(
        #     [custom_transform.CLAHEPreprocess(), custom_transform.ResizeWithPaddingAndBBox()]
        # )

        # Subscribers and Publishers:
        # - Subscribing to the incoming image.
        # - Publishing:
        #   - the detected gauge image (bounding box on needle),
        #   - the processed image (bounding box on needle),
        #   - and the gauge reading.
        try:
            self._image_sub = self.create_subscription(
                Image,
                self._image_topic,
                self.image_callback,
                qos_profile=image_qos_profile,
            )
        except Exception as e:
            self.get_logger().error(f'Failed to create image subscriber: {e}')
            raise

        self._gauge_pub = self.create_publisher(
            Image,
            'gauge_image',
            10,
            # qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        self._proc_gauge_pub = self.create_publisher(
            Image,
            'processed_gauge_image',
            10,
            # qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        self._gauge_reading_pub = self.create_publisher(
            GaugeReading,
            'gauge_reading',
            10,
            # qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        # Service to define how many images are processed
        self._process_mode = GaugeProcess.Request.MODE_CONTINUOUS_PROCESSING
        self._image_process_mode_srv = self.create_service(
            GaugeProcess, 'set_image_process_mode', self.set_image_process_mode_callback
        )

        # cv_bridge for image conversion.
        self._bridge = CvBridge()

        self.get_logger().info('GaugeReader Node Started')

    def detect(self, cv_image):
        _, buf = cv2.imencode('.jpg', cv_image)
        b64 = base64.b64encode(buf).decode('utf-8')
        headers = {'Authorization': f'Bearer {self.token}'}
        r = requests.post(
            self.model_server_url + '/detect', json={'image': b64}, headers=headers
        )
        if r.status_code == 200:
            return r.json()
        else:
            self.get_logger().error(
                f'Error while calling detect: {r.status_code} - {r.text}'
            )
            raise Exception(f'Error while calling detect: {r.status_code} - {r.text}')

    def read(self, crop, bbox):
        self.get_logger().info(f'Calling read on bbox: {bbox}')
        self.get_logger().info(f'Crop shape: {crop.shape}')
        _, buf = cv2.imencode('.jpg', crop)
        b64 = base64.b64encode(buf).decode('utf-8')
        headers = {'Authorization': f'Bearer {self.token}'}
        r = requests.post(
            self.model_server_url + '/read',
            json={'image': b64, 'bbox': bbox},
            headers=headers,
        )

        if r.status_code == 200:
            return r.json().get('reading', 0.0)
        else:
            self.get_logger().error(
                f'Error while calling read: {r.status_code} - {r.text}'
            )
            raise Exception(f'Error while calling detect: {r.status_code} - {r.text}')

    def image_callback(self, msg):
        self.get_logger().info('Received image for processing.')
        if self._process_mode == GaugeProcess.Request.MODE_DO_NOTHING:
            return
        elif self._process_mode == GaugeProcess.Request.MODE_PROCESS_ONE_IMAGE:
            self._process_mode = GaugeProcess.Request.MODE_DO_NOTHING

        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        detection_result = self.detect(cv_image)

        # Do some checks on the detection results
        if detection_result['gauge']['score'] < self._min_gauge_score:
            self.get_logger().warning(
                'Skipping observation: missing valid gauge detection.'
            )
            return
        if detection_result['needle']['score'] < self._min_needle_score:
            self.get_logger().warning(
                'Skipping observation: missing valid needle detection.'
            )
            return
        if not self._needle_in_gauge(
            detection_result['gauge']['bbox'], detection_result['needle']['bbox']
        ):
            self.get_logger().warning('Skipping observation: needle not in gauge.')
            return
        header = msg.header
        cropped_gauge, needle_bbox = self._crop_gauge(cv_image, detection_result)
        self._publish_gauge_image(cropped_gauge.copy(), needle_bbox, header)

        if self._use_math_reading:
            result = self._calculate_gauge_value(cropped_gauge, needle_bbox)

            self.get_logger().info(
                f'Needle bounding box: {needle_bbox} '
                f'(width={needle_bbox[2] - needle_bbox[0]}, '
                f'height={needle_bbox[3] - needle_bbox[1]})'
            )
            self.get_logger().info(f'Needle tip detected at: {result["needle_tip"]}')
            self.get_logger().info(f'Gauge center: {result["center"]}')
            self.get_logger().info(
                f'Gauge Needle Angle: {result["angle_degrees"]:.2f}° of {270}°'
            )
            self.get_logger().info(f'Gauge Value: {result["gauge_value"]:.2f} of 10')

            gauge_reading = result['raw_value']
        else:
            trans_gauge, trans_needle = self._transform_data(cropped_gauge, needle_bbox)
            self._publish_transformed_image(trans_gauge.copy(), trans_needle, header)

            gauge_reading = self.read(trans_gauge, trans_needle)

        # Publish the gauge reading
        self._publish_gauge_reading(gauge_reading, header)


def main(args=None):
    rclpy.init(args=args)
    node = GaugeReaderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
