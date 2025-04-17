FROM ros:humble-ros-base

# Install basic requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    portaudio19-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    software-properties-common \
    ros-humble-rmw-cyclonedds-cpp

# Install specific numpy version first


# Create workspace
WORKDIR /ros2_ws

# Clone the repository with submodules
COPY requirements.txt /ros2_ws/src/requirements.txt
RUN pip install 'numpy<2.0.0'

# Install Python requirements (with numpy constraint)
RUN cd src && pip install -r requirements.txt

# Copy rest of the source code
COPY . /ros2_ws/src

# Install ROS dependencies
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build

# Source ROS2 and workspace in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Remove apt cache
RUN rm -rf /var/lib/apt/lists/*

# add /ros2_ws to the entrypoint
RUN sed --in-place --expression \
      '$isource "/ros2_ws/install/setup.bash"' \
      /ros_entrypoint.sh

ENTRYPOINT [ "/ros_entrypoint.sh" ]


