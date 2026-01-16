#!/bin/bash

source /opt/ros/$ROS_DISTRO/setup.sh
source install/setup.sh

ros2 launch gauge_net gauge_net_lite.launch.py use_math:=$USE_MATH \
     model_server_url:=$MODEL_SERVER_URL joy_enable_button:=$JOY_ENABLE_BUTTON \
     joy_linear_axis:=$JOY_LINEAR_AXIS joy_angular_axis:=$JOY_ANGULAR_AXIS

