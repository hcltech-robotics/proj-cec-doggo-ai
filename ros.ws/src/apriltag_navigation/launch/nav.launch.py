# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    # Get the package share directory
    package_dir = get_package_share_directory('apriltag_navigation')

    # Get the config file
    config_file = os.path.join(
        package_dir, 'config', 'apriltag_controller_params.yaml')

    rectify_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        name='rectify',
        namespace='apriltag',
        parameters=[{
            'output_width': 1920,
            'output_height': 1080,
        }],
        remappings=[('image_raw', '/camera/camera/color/image_raw'),
                    ('camera_info', '/camera/camera/color/camera_info')]
    )

    # resize_node = ComposableNode(
    #     package='isaac_ros_image_proc',
    #     plugin='nvidia::isaac_ros::image_proc::ResizeNode',
    #     name='resize_for_visualization',
    #     namespace='',
    #     parameters=[{
    #         'output_width': 640,
    #         'output_height': 360,  # Maintaining the 16:9 aspect ratio
    #         'keep_aspect_ratio': True
    #     }],
    #     remappings=[('/image', '/camera/camera/color/image_raw'),
    #                 ('/camera_info', '/camera/camera/color/camera_info'),
    #                 ('/resize/image', '/camera/camera/color_resized/image_raw')]
    # )

    apriltag_node = ComposableNode(
        package='isaac_ros_apriltag',
        plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
        name='apriltag',
        namespace='apriltag',
        parameters=[{
            'size': 0.08,
            'max_tags': 32,
        }],
        remappings=[
            ('image', 'image_rect'),
            ('camera_info', 'camera_info_rect'),
            ('tf', '/tf')
        ]
    )

    controller_node = Node(
        package='apriltag_navigation',
        executable='apriltag_controller',
        name='apriltag_controller',
        parameters=[config_file],
        output='screen'
    )

    apriltag_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='apriltag_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[
            rectify_node,
            # resize_node,
            apriltag_node,
        ],
        output='screen'
    )

    return launch.LaunchDescription([apriltag_container, controller_node])
