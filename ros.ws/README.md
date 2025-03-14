# ROS 2 Gauge Network Endpoints

This README provides instructions for building and running the ROS 2 node related to gauge detection and reading.

## Build Instructions

1. **Download the model files** (required before building):  
   ```bash
   ./download_checkpoints.sh
   ```
2. **Build the package with symlink-install:**  
   ```bash
   colcon build --symlink-install
   ```
3. **Source the workspace:**  
   ```bash
   source ./install/setup.sh
   ```

## Prerequisites

Before running the node, ensure the following:

1. Your ROS 2 workspace is sourced.
2. The required model files are available at the expected locations (inside the `models/` directory in the shared folder).

## Available Node

### Gauge Reading Node

The `gauge_reading` node subscribes to an image topic and detects gauges using a trained model. 

#### **Running the Gauge Reader**
```bash
ros2 run gauge_reading gauge_reader
```

#### **Subscribed Topics:**
- **`image_topic`** (`sensor_msgs/msg/Image`) - Input image stream.

#### **Published Topics:**
- **`/gauge_reading/gauge_image`** (`sensor_msgs/msg/Image`) - Detected, cropped image.
- **`/gauge_reading/gauge_preprocessed`** (`sensor_msgs/msg/Image`) - Preprocessed image (if `use_math` is false).
- **`/gauge_reading/gauge_reading`** (`gauge_net_msgs/msg/GaugeReading`) - The final needle reading.

## Running with a Launch File

To start the **gauge reading** node using a launch file, use:

```bash
ros2 launch gauge_reading gauge_net.launch.py
```

### Launch File Arguments:
- **`gauge_detector_weights`** - Path to a `.pt` file for a custom detection model (default: `models/gauge_detect.pt` in the shared folder).
- **`gauge_reader_weights`** - Path to a `.pt` file for a custom reader model (default: `models/gauge_reader.pt` in the shared folder).
- **`image_topic`** - A topic with `sensor_msgs/Image` to change the camera feed (default: `/apriltag/image_rect`).
- **`use_math`** - If set, enables mathematical computation instead of CNN for gauge value reading (default: `true`).

## Setting Image Process Mode

To call the `set_image_process_mode` service, use the following command:
```bash
ros2 service call /gauge_reader/set_image_process_mode gauge_net_interface/srv/GaugeProcess "{process_mode: 0}"
```
The values for `process_mode` can be found in the [GaugeProcess.srv](https://github.com/hcltech-robotics/proj-cec-doggo-ai/blob/main/ros.ws/src/gauge_net/gauge_net_interface/srv/GaugeProcess.srv) file.

This updated README ensures accuracy and provides a clearer understanding of the system. Let me know if further refinements are needed!

