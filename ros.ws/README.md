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
ros2 run gauge_net gauge_detector --ros-args -p model_file:=../checkpoints/gauge_detect.pt
```

### Gauge Reader

Run the following command to launch the gauge reader:
```
ros2 run gauge_net gauge_reader --ros-args -p model_file:=../checkpoints/gauge_net_ResidualSEBlock_with_boxes.pt
```

## Running Both Nodes with a Launch File  

To start both the gauge detector and gauge reader nodes using a single launch file, use:  
```bash
ros2 launch gauge_net gauge_net.launch.py \
    gauge_detector_weights:=/path/to/gauge_detect.pt \
    gauge_reader_weights:=/path/to/gauge_net_ResidualSEBlock_with_boxes.pt
```

