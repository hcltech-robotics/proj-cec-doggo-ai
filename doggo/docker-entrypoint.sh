#!/bin/bash

cd ~

export ACCEPT_EULA=Y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib

export STANDALONE=False

export EXEC_CMD="/app/quadruped.py --quadruped $QUADRUPED --env $ENV $ADDITIONAL_PARAMS"


exec /isaac-sim/runheadless.sh --exec "${EXEC_CMD}"
