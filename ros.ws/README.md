# ROS2 Gauge Network Endpoints



This README provides instructions for building and running the available ROS2 endpoints related to gauge detection and reading.

## Build

1. `colcon build`
1. `source ./install/setup.sh`

## Prerequisites

1. Ensure that you have sourced your ROS2 workspace.
2. Verify that the model files exist at the specified path: run [`download_checkpoints.sh`](../download_checkpoints.sh) 

## Available Endpoints

### Gauge Detector

The gauge detector endpoint receives an image from the `image` topic and runs a specially trained FastRCNN model that detect manual gauges and manual gauge needles. 

Run the following command to launch the gauge detector:
```
ros2 run gauge_net gauge_detector --ros-args -p model_file:=../checkpoints/gauge_detect2.pt
```

The gauge detector sends bounding boxes as a `vision_msgs/msg/Detection2DArray` to the `/detections` topic and detected gauges (image cutouts) to the `/gauge_image` topic.

### Gauge Reader

The gauge reader receives images on the `/image` topic and matching bounding boxes from the `/detection` topic. 

Run the following command to launch the gauge reader:
```
ros2 run gauge_net gauge_reader --ros-args -p model_file:=../checkpoints/gauge_net_with_needle_boxed.pt
```

The gauge reader sends `gauge_net_msgs/msg/GaugeReading` messages to the `/gauge_reading` topic.

