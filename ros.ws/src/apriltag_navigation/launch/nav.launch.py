import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import (
    Node,
    ComposableNodeContainer,
    LoadComposableNodes,
)
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # backend: cpu | gpu
    backend = LaunchConfiguration('backend')

    declare_backend_arg = DeclareLaunchArgument(
        'backend',
        default_value='cpu',
        description='AprilTag detector backend: "cpu" (apriltag_ros) or "gpu" (isaac_ros_apriltag)',
    )

    # Conditions
    use_cpu_backend = PythonExpression([
        "'",
        backend,
        "' == 'cpu'"
    ])

    use_gpu_backend = PythonExpression([
        "'",
        backend,
        "' == 'gpu'"
    ])

    # CPU-based pipeline: image_proc + apriltag_ros
    rectify_cpu_node = ComposableNode(
        package="image_proc",
        plugin="image_proc::RectifyNode",
        name="rectify_cpu",
        namespace="apriltag",
        remappings=[
            ("image", "/camera/image_raw"),
            ("camera_info", "/camera/camera_info"),
            ("image_rect", "/apriltag/image_rect"),
        ],
    )

    apriltag_cpu_node = ComposableNode(
        package="apriltag_ros",
        plugin="AprilTagNode",
        name="apriltag_cpu",
        namespace="apriltag",
        parameters=[
            {
                "family": "36h11",
                "size": 0.08,
            }
        ],
        remappings=[
            ("image_rect", "/apriltag/image_rect"),
            ("camera_info", "/camera/camera_info"),
            ("tf", "/tf"),
        ],
    )

    # GPU-based pipeline: isaac_ros_image_proc + isaac_ros_apriltag
    rectify_gpu_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        name='rectify_gpu',
        namespace='apriltag',
        parameters=[{
            'output_width': 1280,
            'output_height': 720,
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('image_rect', '/apriltag/image_rect'),
        ],
    )

    apriltag_gpu_node = ComposableNode(
        package="isaac_ros_apriltag",
        plugin="nvidia::isaac_ros::apriltag::AprilTagNode",
        name="apriltag_gpu",
        namespace="apriltag",
        parameters=[
            {
                "size": 0.08,
                "max_tags": 32,
            }
        ],
        remappings=[
            ("image", "/apriltag/image_rect"),
            ("camera_info", "/camera/camera_info"),
            ("tf", "/tf"),
        ],
    )

    # Container: empty at start, we load into it dynamically
    apriltag_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="apriltag_container",
        namespace="",
        executable="component_container_mt",
        composable_node_descriptions=[],
        output="screen",
    )

    # Conditionally load CPU nodes
    load_cpu_nodes = LoadComposableNodes(
        target_container='apriltag_container',
        composable_node_descriptions=[
            rectify_cpu_node,
            apriltag_cpu_node,
        ],
        condition=IfCondition(use_cpu_backend),
    )

    # Conditionally load GPU nodes
    load_gpu_nodes = LoadComposableNodes(
        target_container='apriltag_container',
        composable_node_descriptions=[
            rectify_gpu_node,
            apriltag_gpu_node,
        ],
        condition=IfCondition(use_gpu_backend),
    )

    # Controller node config
    config_file = PathJoinSubstitution([
        FindPackageShare('apriltag_navigation'),
        'config',
        'apriltag_controller_params.yaml',
    ])

    controller_node = Node(
        package="apriltag_navigation",
        executable="apriltag_controller",
        name="apriltag_controller",
        parameters=[config_file],
        output="screen",
    )

    return LaunchDescription([
        declare_backend_arg,
        apriltag_container,
        load_cpu_nodes,
        load_gpu_nodes,
        controller_node,
    ])
