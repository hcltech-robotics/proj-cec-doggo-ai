import rclpy
from rclpy.node import Node
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gauge_net_interface.msg import GaugeReading


class GaugeReader(Node):
    def __init__(self):
        super().__init__('resnet_image_processor')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Declare and get model file parameter
        self.declare_parameter('model_file', '')
        self.declare_parameter('scaling_min', 0.0)
        self.declare_parameter('scaling_max', 100.0)
        model_path = self.get_parameter('model_file').get_parameter_value().string_value
        self.scaling_min = self.get_parameter('scaling_min').get_parameter_value().double_value
        self.scaling_max = self.get_parameter('scaling_max').get_parameter_value().double_value
        
        # Load ResNet model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Image processing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        # ROS2 Image subscriber and publisher
        self.image_sub = self.create_subscription(Image, 'processed_image', self.image_callback, 10)
        self.reading_pub = self.create_publisher(GaugeReading, 'gauge_reading', 10)

        self.bridge = CvBridge()
        self.get_logger().info("ResNet Image Processor Node Started")

    def image_callback(self, msg):
        # Convert ROS2 image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # Convert image to tensor
        image_tensor = self.transform(cv_image).unsqueeze(0)  # Add batch dimension
        print(image_tensor)
        image_tensor = image_tensor.to(self.device)
        # Run inference
        with torch.no_grad():
            reading = self.model(image_tensor).item()
        print(reading)
        scaled_reading = self.scaling_min + reading * (self.scaling_max - self.scaling_min)
        gauge_reading = GaugeReading()
        gauge_reading.reading = reading
        gauge_reading.scaled_reading = scaled_reading
        print(f"Reading: {reading}, Scaled Reading: {scaled_reading}")
        self.reading_pub.publish(gauge_reading)

def main(args=None):
    rclpy.init(args=args)
    node = GaugeReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
