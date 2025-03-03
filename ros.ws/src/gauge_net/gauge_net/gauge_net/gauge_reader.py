from cv_bridge import CvBridge
from gauge_net_interface.msg import GaugeReading

# Import message_filters for synchronization.
from message_filters import Subscriber, TimeSynchronizer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
import torchvision.transforms as transforms
from vision_msgs.msg import Detection2DArray


class GaugeReader(Node):

    def __init__(self):
        super().__init__('gauge_reader')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_input_size = (512, 512)

        # Declare parameters.
        self.declare_parameter('model_file', '')
        self.declare_parameter('min_score', 0.99)
        self.declare_parameter('scaling_min', 0.0)
        self.declare_parameter('scaling_max', 100.0)
        model_path = self.get_parameter('model_file').get_parameter_value().string_value
        self.scaling_min = self.get_parameter('scaling_min').get_parameter_value().double_value
        self.scaling_max = self.get_parameter('scaling_max').get_parameter_value().double_value
        self.min_score = self.get_parameter('min_score').get_parameter_value().double_value

        # Load the model (expects inputs: image tensor and bbox tensor).
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Define image processing pipeline.
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.model_input_size),
                transforms.Grayscale(num_output_channels=1),  # Converts to grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        # Publisher for the gauge reading.
        self.reading_pub = self.create_publisher(GaugeReading, 'gauge_reading', 10)

        # cv_bridge for image conversion.
        self.bridge = CvBridge()

        # Create message_filters subscribers for gauge_image and detections.
        self.gauge_sub = Subscriber(self, Image, 'image')
        self.detections_sub = Subscriber(self, Detection2DArray, 'detections')

        # Use a TimeSynchronizer to match messages based on their header timestamps.
        self.ts = TimeSynchronizer([self.gauge_sub, self.detections_sub], 10)
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info('GaugeReader Node Started')

    def synced_callback(self, image_msg, detections_msg):
        """
        Process synchronized image and detection messages.

        This callback is triggered only when both `image_msg` and `detections_msg`
        share the same header timestamp. It performs the following steps:

        1. Converts the image to a tensor.
        2. Extracts the gauge (where `class_id == '2'`).
        3. Extracts a normalized bounding box for the needle from detections (`class_id == '1'`).
        4. Passes both to the model to compute the gauge reading.
        """
        # Convert the image message to an OpenCV image.
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        # print(detections_msg)
        # Search for valid gauge and needle detections.
        gauge_det = None
        needle_det = None
        for det in detections_msg.detections:
            for hyp in det.results:
                # Ensure hypothesis is an ObjectHypothesisWithPose with a string class_id.
                if hyp.hypothesis.class_id == '1' and hyp.hypothesis.score >= self.min_score:
                    gauge_det = det
                elif hyp.hypothesis.class_id == '2' and hyp.hypothesis.score >= self.min_score:
                    needle_det = det

        # If either detection is missing, skip this observation.
        print(gauge_det, needle_det)
        if gauge_det is None or needle_det is None:
            self.get_logger().warning(
                'Skipping observation: missing valid gauge/needle detection or confidence < 95%.'
            )
            return

        # Extract gauge bounding box.
        g_box = gauge_det.bbox
        g_center_x = g_box.center.position.x
        g_center_y = g_box.center.position.y
        g_size_x = g_box.size_x
        g_size_y = g_box.size_y
        g_x_min = int(g_center_x - g_size_x / 2.0)
        g_y_min = int(g_center_y - g_size_y / 2.0)
        g_x_max = int(g_center_x + g_size_x / 2.0)
        g_y_max = int(g_center_y + g_size_y / 2.0)

        # Check that the gauge bounding box is within the image bounds.
        height, width, _ = cv_image.shape
        if g_x_min < 0 or g_y_min < 0 or g_x_max > width or g_y_max > height:
            self.get_logger().warning(
                'Gauge bounding box is out of image bounds, skipping observation.'
            )
            return

        # Crop the gauge from the image.
        gauge_crop = cv_image[g_y_min:g_y_max, g_x_min:g_x_max]

        # Extract needle bounding box.
        n_box = needle_det.bbox
        n_center_x = n_box.center.position.x
        n_center_y = n_box.center.position.y
        n_size_x = n_box.size_x
        n_size_y = n_box.size_y

        # Compute the needle bounding box in the gauge crop.
        n_x_min = n_center_x - n_size_x / 2.0
        n_y_min = n_center_y - n_size_y / 2.0
        n_x_max = n_center_x + n_size_x / 2.0
        n_y_max = n_center_y + n_size_y / 2.0

        # Ensure that the needle box is fully inside the gauge box.
        print('needle:', n_x_min, n_y_min, n_x_max, n_y_max)
        print('gauge:', g_x_min, g_y_min, g_x_max, g_y_max)
        if n_x_min < g_x_min or n_y_min < g_y_min or n_x_max > g_x_max or n_y_max > g_y_max:
            self.get_logger().warning(
                'Needle bounding box is not fully inside the gauge, skipping observation.'
            )
            return

        # Normalize the needle bbox relative to the gauge crop.
        gauge_width = g_x_max - g_x_min
        gauge_height = g_y_max - g_y_min

        norm_n_x_min = (n_x_min - g_x_min) / gauge_width
        norm_n_y_min = (n_y_min - g_y_min) / gauge_height
        norm_n_x_max = (n_x_max - g_x_min) / gauge_width
        norm_n_y_max = (n_y_max - g_y_min) / gauge_height

        norm_n_x_min *= self.model_input_size[0] / gauge_width
        norm_n_y_min *= self.model_input_size[1] / gauge_height
        norm_n_x_max *= self.model_input_size[0] / gauge_width
        norm_n_y_max *= self.model_input_size[1] / gauge_height

        bbox_tensor = torch.tensor(
            [norm_n_x_min, norm_n_y_min, norm_n_x_max, norm_n_y_max], dtype=torch.float32
        ).unsqueeze(0)
        # Transform the gauge crop to a tensor (model expects 512x512 input).
        gauge_crop_tensor = self.transform(gauge_crop).unsqueeze(0).to(self.device)

        # Run inference: the model expects the gauge crop and the normalized bbox.
        with torch.no_grad():
            reading = self.model(gauge_crop_tensor, bbox_tensor.to(self.device)).item()
            # reading = self.model(gauge_crop_tensor).item()

        # Scale the model output.
        scaled_reading = self.scaling_min + reading * (self.scaling_max - self.scaling_min)

        # Populate and publish the gauge reading message.
        gauge_reading = GaugeReading()
        gauge_reading.reading = reading
        gauge_reading.scaled_reading = scaled_reading
        self.get_logger().info(
            f'BBOX: {bbox_tensor} Reading: {reading}, Scaled Reading: {scaled_reading}'
        )

        self.reading_pub.publish(gauge_reading)


def main(args=None):
    rclpy.init(args=args)
    node = GaugeReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
