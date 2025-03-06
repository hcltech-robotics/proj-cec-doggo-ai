from cv_bridge import CvBridge
from gauge_net_interface.srv import GaugeProcess
import rclpy
from rclpy.node import Node
from rclpy.qos_overriding_options import QoSOverridingOptions
from sensor_msgs.msg import Image
import torch
import torchvision.transforms as transforms
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)


class GaugeDetector(Node):

    def __init__(self):
        super().__init__('gauge_detecor')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare and get model file parameter
        self.declare_parameter('model_file', '')
        self.declare_parameter('min_gauge_score', 0.99)
        model_path = self.get_parameter('model_file').get_parameter_value().string_value
        self.min_gauge_score = (
            self.get_parameter('min_gauge_score').get_parameter_value().double_value
        )

        # Load ResNet model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Image processing pipeline
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Subscribers and Publishers:
        # - Subscribing to the incoming image.
        # - Publishing the gauge image (as is) and the Detection2DArray message.
        self.image_sub = self.create_subscription(
            Image,
            'image',
            self.image_callback,
            1,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )
        self.gauge_pub = self.create_publisher(
            Image,
            'gauge_image',
            10,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            'detections',
            10,
            qos_overriding_options=QoSOverridingOptions.with_default_policies(),
        )

        # ROS2 Service to define how many images are processed
        self.process_mode_ = GaugeProcess.Request.MODE_DO_NOTHING
        self.image_process_mode_srv_ = self.create_service(
            GaugeProcess, 'set_image_process_mode', self.set_image_process_mode_callback
        )

        self.bridge = CvBridge()
        self.get_logger().info('GaugeDetector Node Started')

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
            self.process_mode_ = request.process_mode
            self.get_logger().debug(f'Set to {mode_name}')
            response.success = True
            response.info = f'Set to {mode_name}'
        else:
            self.get_logger().warning(f'Invalid process mode: {request.process_mode}')
            response.success = False
            response.info = f'Invalid process mode ({request.process_mode})'

        return response

    def image_callback(self, msg):

        if self.process_mode_ == GaugeProcess.Request.MODE_DO_NOTHING:
            return
        elif self.process_mode_ == GaugeProcess.Request.MODE_PROCESS_ONE_IMAGE:
            self.process_mode_ = GaugeProcess.Request.MODE_DO_NOTHING

        # Process the image for detection
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        image_tensor = self.transform(cv_image)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            detections = self.model([image_tensor])

        # (2) Create a Detection2DArray message for bounding boxes.
        detections_msg = Detection2DArray()
        # Use the same header (and timestamp) as the incoming image so that
        # a receiver can perform exact time synchronization.
        detections_msg.header = msg.header
        for (detection, label_idx), score in zip(
            zip(detections[1][0]['boxes'].cpu(), detections[1][0]['labels'].cpu()),
            detections[1][0]['scores'].cpu(),
        ):

            box = detection.numpy().astype(int)
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])

            if label_idx.item() == 1:
                self.process_gauge_detection(x_min, y_min, x_max, y_max, score, msg, cv_image)

            detection = Detection2D()
            detection.header = msg.header

            # Compute the center and size of the bounding box.
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            detection.bbox.center.position.x = float(center_x)
            detection.bbox.center.position.y = float(center_y)
            detection.bbox.size_x = float(x_max - x_min)
            detection.bbox.size_y = float(y_max - y_min)
            object_hypothesis = ObjectHypothesis(
                class_id=str(label_idx.item()), score=score.item()
            )
            object_hypothesis_with_pose = ObjectHypothesisWithPose(hypothesis=object_hypothesis)
            detection.results.append(object_hypothesis_with_pose)
            detections_msg.detections.append(detection)
            self.get_logger().info(f'Detected bounding box: {box}')

        # Publish the Detection2DArray message
        self.detections_pub.publish(detections_msg)

    def process_gauge_detection(self, x_min, y_min, x_max, y_max, score, msg, cv_image):
        if score > self.min_gauge_score:
            cropped_image = cv_image[y_min:y_max, x_min:x_max]
            gauge_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding='rgb8')
            gauge_msg.header = msg.header  # Maintain the original header for synchronization
            self.gauge_pub.publish(gauge_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GaugeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
