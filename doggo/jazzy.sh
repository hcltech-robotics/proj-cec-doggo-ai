#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYVER="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
export ROS_DISTRO=jazzy

# If ros2 startup fails to use cyclone, use the following 2 commands to install cyclone dds
# sudo apt update
# sudo apt install ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python$PYVER/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib 

"$@"