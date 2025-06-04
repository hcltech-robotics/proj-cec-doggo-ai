FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ros_ws

RUN rm -f /usr/share/keyrings/ros-archive-keyring.gpg \
    && rm -f /etc/apt/sources.list.d/ros2-latest.list

# Install necessary tools
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release

# Install the new ROS 2 apt source package
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2-latest.list

# Now install pre-reqs and your packages
RUN apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp \
    python3-pip \
    python3-colcon-common-extensions \
    python3-dev \
    git

# Copy workspace and requirements.txt
COPY ros.ws ./src
COPY requirements.txt ./requirements.txt

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch CPU wheels manually 
RUN pip3 install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21 \
    --index-url https://download.pytorch.org/whl/cpu

# Install ROS dependencies and build
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build

# Clean up apt cache
RUN rm -rf /var/lib/apt/lists/*

# Add entrypoint script
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
