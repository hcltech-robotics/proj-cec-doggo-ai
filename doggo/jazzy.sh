#!/bin/bash

export PYVER="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python$PYVER/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib 

"$@"