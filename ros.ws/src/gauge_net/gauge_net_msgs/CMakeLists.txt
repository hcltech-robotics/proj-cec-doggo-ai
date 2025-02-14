cmake_minimum_required(VERSION 3.8)
project(gauge_net_msgs)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GaugeReading.msg"
  DEPENDENCIES std_msgs
)

ament_package()
