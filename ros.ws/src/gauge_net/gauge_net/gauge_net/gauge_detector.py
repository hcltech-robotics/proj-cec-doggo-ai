import rclpy
from rclpy.node import Node
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GaugeDetector(Node):
    def __init__(self):
        super().__init__('GaugeDetector')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Declare and get model file parameter
        self.declare_parameter('model_file', '')
        model_path = self.get_parameter('model_file').get_parameter_value().string_value
        
        # Load ResNet model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Image processing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        # ROS2 Image subscriber and publisher
        self.image_sub = self.create_subscription(Image, 'image', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)

        self.bridge = CvBridge()
        self.get_logger().info("GaugeDetector Node Started")

    def image_callback(self, msg):
        # Convert ROS2 image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # Convert image to tensor
        image_tensor = self.transform(cv_image)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        # Run inference
        with torch.no_grad():
            detections = self.model([image_tensor])

        # Assume label 1 is our target class (modify as needed)
        detections = self.parse_detections(detections)

        for detection in detections: #we usually get 2
            self.get_logger().info(f"Detected bounding box: {detection}")

            # Crop the first detected region
            x_min, y_min, x_max, y_max = detection
            cropped_image = cv_image[y_min:y_max, x_min:x_max]

            # Convert back to ROS Image and publish
            processed_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding='rgb8')
            self.image_pub.publish(processed_msg)

    def parse_detections(self, detections):
        """ Parse outputs and extract bounding boxes for label 1. """
        coordinates = []
        for ((detection, label_idx), score) in zip(zip(detections[1][0]['boxes'].cpu(), detections[1][0]['labels'].cpu()), detections[1][0]['scores'].cpu()):
            if label_idx.item() != 1:
                continue
            box = detection.numpy().astype(int)
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])

            if score > 0.75:
                coordinates.append((x_min, y_min, x_max, y_max))
        return coordinates


def main(args=None):
    rclpy.init(args=args)
    node = GaugeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
