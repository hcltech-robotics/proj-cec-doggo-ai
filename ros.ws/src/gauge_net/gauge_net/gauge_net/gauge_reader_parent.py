from abc import ABC, abstractmethod
import cv2
import math
import numpy as np
from rclpy.node import Node
from gauge_net_interface.msg import GaugeReading
from gauge_net_interface.srv import GaugeProcess


class GaugeReaderParent(Node, ABC):
    @abstractmethod
    def detect(self, cv_image):
        pass

    @abstractmethod
    def read(self, crop, bbox):
        pass

    def set_image_process_mode_callback(
        self, request: GaugeProcess.Request, response: GaugeProcess.Response
    ) -> GaugeProcess.Response:
        PROCESS_MODE_NAMES = {
            GaugeProcess.Request.MODE_DO_NOTHING: "MODE_DO_NOTHING",
            GaugeProcess.Request.MODE_PROCESS_ONE_IMAGE: "MODE_PROCESS_ONE_IMAGE",
            GaugeProcess.Request.MODE_CONTINUOUS_PROCESSING: "MODE_CONTINUOUS_PROCESSING",
        }

        mode_name = PROCESS_MODE_NAMES.get(request.process_mode)

        if mode_name:
            self._process_mode = request.process_mode
            self.get_logger().debug(f"Set to {mode_name}")
            response.success = True
            response.info = f"Set to {mode_name}"
        else:
            self.get_logger().warning(f"Invalid process mode: {request.process_mode}")
            response.success = False
            response.info = f"Invalid process mode ({request.process_mode})"

        return response

    def _needle_in_gauge(self, gauge_bbox, needle_bbox):
        if gauge_bbox is None or needle_bbox is None:
            return False

        # Unpack bounding box coordinates [x1, y1, x2, y2]
        g_x1, g_y1, g_x2, g_y2 = gauge_bbox
        n_x1, n_y1, n_x2, n_y2 = needle_bbox

        # Check if needle is completely inside gauge
        return g_x1 <= n_x1 and n_x2 <= g_x2 and g_y1 <= n_y1 and n_y2 <= g_y2

    def _crop_gauge(self, cv_image, detection_result):
        gauge_bbox = detection_result["gauge"]["bbox"]
        needle_bbox = detection_result["needle"]["bbox"]

        # Convert tensor values to integers
        if gauge_bbox is not None:
            gauge_bbox = [int(coord) for coord in gauge_bbox]

        if needle_bbox is not None:
            needle_bbox = [int(coord) for coord in needle_bbox]

        # Extract gauge bounding box.
        x_min, y_min, x_max, y_max = gauge_bbox
        cropped_gauge = cv_image[y_min:y_max, x_min:x_max]

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
            self._bridge.cv2_to_imgmsg(cropped_gauge, encoding="rgb8", header=header)
        )

    def _transform_data(self, cropped_gauge, needle_bbox):
        # Transform the image and needle bounding box
        height, width, _ = cropped_gauge.shape
        needle_bbox = np.array(needle_bbox) / np.array([width, height, width, height])
        sample = {"image": cropped_gauge, "bbox": needle_bbox}
        transformed = self._reader_transform(sample)

        # Return both image and bounding box as numpy arrays
        return np.array(transformed["image"]), transformed["bbox"]

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
            self._bridge.cv2_to_imgmsg(gauge_image_rgb, encoding="rgb8", header=header)
        )

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

        self.get_logger().info(
            f"Gauge reading: {gauge_reading}, Scaled reading: {scaled_reading}"
        )

        self._gauge_reading_pub.publish(gauge_reading_msg)

    def _calculate_gauge_value(self, cropped_gauge, needle_bbox):
        # Image dimensions
        height, width = cropped_gauge.shape[:2]

        # Gauge center
        C_x, C_y = width / 2, height / 2

        # Extract needle region
        x_min, y_min, x_max, y_max = needle_bbox

        test_points = [
            (x_min, y_min),
            (x_max, y_min),  # Top corners
            (x_min, y_max),
            (x_max, y_max),  # Bottom corners
            ((x_min + x_max) / 2, y_min),  # Top middle
            ((x_min + x_max) / 2, y_max),  # Bottom middle
            (x_min, (y_min + y_max) / 2),  # Left middle
            (x_max, (y_min + y_max) / 2),  # Right middle
        ]
        N_x, N_y = max(test_points, key=lambda p: math.dist(p, (C_x, C_y)))

        # Compute vector from center to needle tip
        V_x = N_x - C_x  # Horizontal distance from center
        V_y = C_y - N_y  # Vertical distance (flipped for image coords)

        # Calculate the angle relative to the vertical dividing line
        theta_rad = math.atan2(V_x, V_y)  # Swapped args to align with vertical axis
        theta_deg = math.degrees(theta_rad)

        # Ensure angle is within [-135°, 135°]
        if theta_deg < -135:
            theta_deg += 360  # Wrap around if needed

        # Convert to gauge value (0-10 range)
        gauge_value = (theta_deg + 135) / 270 * self._scaling_max

        return {
            "needle_tip": (N_x, N_y),
            "center": (C_x, C_y),
            "angle_degrees": theta_deg,
            "gauge_value": gauge_value,
            "raw_value": gauge_value / self._scaling_max,  # Normalized value
        }
