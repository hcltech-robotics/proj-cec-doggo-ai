#!/usr/bin/env python3

import os

import cv2
from cv_bridge import CvBridge
from gauge_net.transforms import custom_transform
from gauge_net_interface.msg import GaugeReading
from gauge_net_interface.srv import GaugeProcess
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy
from rclpy.qos_overriding_options import QoSOverridingOptions, QoSProfile
from sensor_msgs.msg import Image
import torch
import torchvision.transforms as transforms


class GaugeReaderNode(Node):

    def __init__(self):
        self._namespace = 'gauge_reader'
        self._node_name = 'gauge_reader'
        super().__init__(self._node_name, namespace=self._namespace)

        # Use the GPU if available.
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare and get node parameters.
        self.declare_parameters(
            namespace=self._namespace,
            parameters=[
                ('image_topic', '/image'),
                ('detector_model_file', ''),
                ('reader_model_file', ''),
                ('min_gauge_score', 0.99),
                ('min_needle_score', 0.95),
                ('scaling_min', 0.0),
                ('scaling_max', 100.0),
                ('image_stream.reliability', rclpy.Parameter.Type.STRING),
                ('image_stream.history', rclpy.Parameter.Type.STRING),
                ('image_stream.depth', rclpy.Parameter.Type.INTEGER),
            ],
        )

        self._image_topic = (
            self.get_parameter(f'{self._node_name}.image_topic').get_parameter_value().string_value
        )
        self._min_gauge_score = (
            self.get_parameter(f'{self._node_name}.min_gauge_score')
            .get_parameter_value()
            .double_value
        )
        self._min_needle_score = (
            self.get_parameter(f'{self._node_name}.min_needle_score')
            .get_parameter_value()
            .double_value
        )
        self._scaling_min = (
            self.get_parameter(f'{self._node_name}.scaling_min').get_parameter_value().double_value
        )
        self._scaling_max = (
            self.get_parameter(f'{self._node_name}.scaling_max').get_parameter_value().double_value
        )
        detector_model_path = (
            self.get_parameter(f'{self._node_name}.detector_model_file')
            .get_parameter_value()
            .string_value
        )
        reader_model_path = (
            self.get_parameter(f'{self._node_name}.reader_model_file')
            .get_parameter_value()
            .string_value
        )
        image_reliability = self.get_parameter(f'{self._node_name}.image_stream.reliability').value
        image_history = self.get_parameter(f'{self._node_name}.image_stream.history').value
        image_depth = self.get_parameter(f'{self._node_name}.image_stream.depth').value

        if not os.path.isfile(detector_model_path):
            self.get_logger().error(f'Detector model file not found: {detector_model_path}')
            raise FileNotFoundError(f'Missing detector model: {detector_model_path}')

        if not os.path.isfile(reader_model_path):
            self.get_logger().error(f'Reader model file not found: {reader_model_path}')
            raise FileNotFoundError(f'Missing reader model: {reader_model_path}')

        # Load the detector model.
        self._detector_model = torch.jit.load(detector_model_path, map_location=self._device)
        self._detector_model.eval()

        # Load the reader model.
        self._reader_model = torch.jit.load(reader_model_path, map_location=self._device)
        self._reader_model.eval()

        # Image processing pipeline for the detector.
        self._detector_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Image processing pipeline for the reader.
        self._reader_transform = transforms.Compose(
            [custom_transform.CLAHEPreprocess(), custom_transform.ResizeWithPaddingAndBBox()]
        )

        # Setting up QoS profile for the image subscriber.
        RELIABILITY_MAP = {
            'best_effort': ReliabilityPolicy.BEST_EFFORT,
            'reliable': ReliabilityPolicy.RELIABLE,
        }

        HISTORY_MAP = {'keep_all': HistoryPolicy.KEEP_ALL, 'keep_last': HistoryPolicy.KEEP_LAST}

        image_qos_profile = QoSProfile(
            reliability=RELIABILITY_MAP.get(image_reliability, ReliabilityPolicy.RELIABLE),
            history=HISTORY_MAP.get(image_history, HistoryPolicy.KEEP_LAST),
            depth=image_depth,
        )

        # Subscribers and Publishers:
        # - Subscribing to the incoming image.
        # - Publishing:
        #   - the detected gauge image (bounding box on needle),
        #   - the processed image (bounding box on needle),
        #   - and the gauge reading.
        try:
            self._image_sub = self.create_subscription(
                Image, self._image_topic, self.image_callback, qos_profile=image_qos_profile
            )
        except Exception as e:
            self.get_logger().error(f'Failed to create image subscriber: {e}')
            raise

        self._gauge_pub = self.create_publisher(
            Image,
            'gauge_image',
            10,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        self._proc_gauge_pub = self.create_publisher(
            Image,
            'processed_gauge_image',
            10,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        self._gauge_reading_pub = self.create_publisher(
            GaugeReading,
            'gauge_reading',
            10,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        # Service to define how many images are processed
        self._process_mode = GaugeProcess.Request.MODE_DO_NOTHING
        self._image_process_mode_srv = self.create_service(
            GaugeProcess, 'set_image_process_mode', self.set_image_process_mode_callback
        )

        # cv_bridge for image conversion.
        self._bridge = CvBridge()

        self.get_logger().info('GaugeReader Node Started')

    def set_image_process_mode_callback(
        self, request: GaugeProcess.Request, response: GaugeProcess.Response
    ) -> GaugeProcess.Response:

        PROCESS_MODE_NAMES = {
            GaugeProcess.Request.MODE_DO_NOTHING: 'MODE_DO_NOTHING',
            GaugeProcess.Request.MODE_PROCESS_ONE_IMAGE: 'MODE_PROCESS_ONE_IMAGE',
            GaugeProcess.Request.MODE_CONTINUOUS_PROCESSING: 'MODE_CONTINUOUS_PROCESSING',
        }

        mode_name = PROCESS_MODE_NAMES.get(request.process_mode)

        if mode_name:
            self._process_mode = request.process_mode
            self.get_logger().debug(f'Set to {mode_name}')
            response.success = True
            response.info = f'Set to {mode_name}'
        else:
            self.get_logger().warning(f'Invalid process mode: {request.process_mode}')
            response.success = False
            response.info = f'Invalid process mode ({request.process_mode})'

        return response

    def detect_gauge(self, cv_image):
        # Process the image for the detector model.
        image_tensor = self._detector_transform(cv_image).to(self._device)
        self.get_logger().info(f'Calling detector model with image tensor: {image_tensor.shape}')
        with torch.no_grad():
            detections = self._detector_model([image_tensor])

        # Extract detections, boxes, and scores
        boxes = detections[1][0]['boxes']
        scores = detections[1][0]['scores']
        labels = detections[1][0]['labels']

        best_detection = {
            'gauge': {'bbox': None, 'score': 0.0},
            'needle': {'bbox': None, 'score': 0.0},
        }

        # Find the detection with highest probability (score) for both the needle and the gauge.
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score > best_detection['gauge']['score']:
                best_detection['gauge']['bbox'] = box
                best_detection['gauge']['score'] = score
            elif label == 2 and score > best_detection['needle']['score']:
                best_detection['needle']['bbox'] = box
                best_detection['needle']['score'] = score

        return best_detection

    def _needle_in_gauge(self, gauge_bbox, needle_bbox):
        if gauge_bbox is None or needle_bbox is None:
            return False

        # Unpack bounding box coordinates [x1, y1, x2, y2]
        g_x1, g_y1, g_x2, g_y2 = gauge_bbox
        n_x1, n_y1, n_x2, n_y2 = needle_bbox

        # Check if needle is completely inside gauge
        return g_x1 <= n_x1 and n_x2 <= g_x2 and g_y1 <= n_y1 and n_y2 <= g_y2

    def _crop_gauge(self, cv_image, detection_result):
        gauge_bbox = detection_result['gauge']['bbox']
        needle_bbox = detection_result['needle']['bbox']

        # Convert tensor values to integers
        if gauge_bbox is not None:
            gauge_bbox = [int(coord) for coord in gauge_bbox]

        if needle_bbox is not None:
            needle_bbox = [int(coord) for coord in needle_bbox]

        # Extract gauge bounding box.
        cropped_gauge = cv_image[gauge_bbox[1]:gauge_bbox[3], gauge_bbox[0]:gauge_bbox[2]]
        # Compute the needle bounding box in the gauge crop.
        needle_x_min = needle_bbox[0] - gauge_bbox[0]
        needle_y_min = needle_bbox[1] - gauge_bbox[1]
        needle_x_max = needle_bbox[2] - gauge_bbox[0]
        needle_y_max = needle_bbox[3] - gauge_bbox[1]

        return cropped_gauge, [needle_x_min, needle_y_min, needle_x_max, needle_y_max]

    def _publish_gauge_image(self, cropped_gauge, needle_bbox, header):
        needle_x_min, needle_y_min, needle_x_max, needle_y_max = needle_bbox
        # Draw rectangle on the needle (red)
        cv2.rectangle(
            cropped_gauge,
            (needle_x_min, needle_y_min),
            (needle_x_max, needle_y_max),
            (255, 0, 0),
            1,
        )
        # Publish the gauge image
        self._gauge_pub.publish(
            self._bridge.cv2_to_imgmsg(cropped_gauge, encoding='rgb8', header=header)
        )

    def _transform_data(self, cropped_gauge, needle_bbox):
        # Transform the image and needle bounding box
        height, width, _ = cropped_gauge.shape
        needle_bbox = np.array(needle_bbox) / np.array([width, height, width, height])
        sample = {'image': cropped_gauge, 'bbox': needle_bbox}
        transformed = self._reader_transform(sample)

        # Return both image and bounding box as numpy arrays
        return np.array(transformed['image']), transformed['bbox']

    def _publish_transformed_image(self, gauge_image, needle_bbox, header):
        # Scale the bounding box back to the image dimensions
        height, width = gauge_image.shape
        needle_bbox = np.array(needle_bbox) * np.array([width, height, width, height])
        needle_bbox = needle_bbox.astype(int)

        # Convert grayscale to RGB
        gauge_image_rgb = cv2.cvtColor(gauge_image, cv2.COLOR_GRAY2RGB)

        # Draw rectangle on the needle (red)
        cv2.rectangle(
            gauge_image_rgb,
            (needle_bbox[0], needle_bbox[1]),
            (needle_bbox[2], needle_bbox[3]),
            (255, 0, 0),  # Red color
            1,
        )

        # Publish the processed image
        self._proc_gauge_pub.publish(
            self._bridge.cv2_to_imgmsg(gauge_image_rgb, encoding='rgb8', header=header)
        )

    def _read_gauge_value(self, gauge_image, needle_bbox):
        # Transform the image to a tensor
        gauge_tensor = transforms.ToTensor()(gauge_image).unsqueeze(0).to(self._device)
        bbox_tensor = torch.tensor(needle_bbox, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Call the reader model
        with torch.no_grad():
            output = self._reader_model(gauge_tensor, bbox_tensor)

        # Get the gauge reading
        gauge_reading = output.item()

        return gauge_reading

    def _publish_gauge_reading(self, gauge_reading, header):
        # Scale the model output
        scaled_reading = self._scaling_min + gauge_reading * (
            self._scaling_max - self._scaling_min
        )

        # Populate and publish the gauge reading message
        gauge_reading_msg = GaugeReading()
        gauge_reading_msg.header = header
        gauge_reading_msg.reading = gauge_reading
        gauge_reading_msg.scaled_reading = scaled_reading

        self.get_logger().info(f'Gauge reading: {gauge_reading}, Scaled reading: {scaled_reading}')

        self._gauge_reading_pub.publish(gauge_reading_msg)

    def image_callback(self, msg):
        if self._process_mode == GaugeProcess.Request.MODE_DO_NOTHING:
            return
        elif self._process_mode == GaugeProcess.Request.MODE_PROCESS_ONE_IMAGE:
            self._process_mode = GaugeProcess.Request.MODE_DO_NOTHING

        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        detection_result = self.detect_gauge(cv_image)

        # Do some checks on the detection results
        if detection_result['gauge']['score'] < self._min_gauge_score:
            self.get_logger().warning('Skipping observation: missing valid gauge detection.')
            return
        if detection_result['needle']['score'] < self._min_needle_score:
            self.get_logger().warning('Skipping observation: missing valid needle detection.')
            return
        if not self._needle_in_gauge(
            detection_result['gauge']['bbox'], detection_result['needle']['bbox']
        ):
            self.get_logger().warning('Skipping observation: needle not in gauge.')
            return
        header = msg.header
        cropped_gauge, needle_bbox = self._crop_gauge(cv_image, detection_result)
        self._publish_gauge_image(cropped_gauge.copy(), needle_bbox, header)

        trans_gauge, trans_needle = self._transform_data(cropped_gauge, needle_bbox)
        self._publish_transformed_image(trans_gauge.copy(), trans_needle, header)

        gauge_reading = self._read_gauge_value(trans_gauge, trans_needle)

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
