#!/bin/bash
set -e

# Source ROS and your workspace
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

# Run the container's main command
exec "$@"
