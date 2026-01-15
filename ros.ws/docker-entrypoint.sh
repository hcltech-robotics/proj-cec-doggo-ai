#!/bin/bash

source /opt/ros/$ROS_DISTRO/setup.sh
source install/setup.sh

ros2 launch gauge_net gauge_net_lite.launch.py use_math:=True model_server_url:=$MODEL_SERVER_URL