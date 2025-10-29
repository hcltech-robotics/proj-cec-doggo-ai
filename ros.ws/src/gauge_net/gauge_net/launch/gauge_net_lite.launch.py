import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
import launch_ros.actions


def generate_launch_description():
    on_exit = LaunchConfiguration('on_exit', default='shutdown')

    package_name = 'gauge_net'

    # Get package share directory
    package_share_dir = get_package_share_directory(package_name)

    default_model_server_url = 'http://localhost:5000'
    default_token = 'doggodoggo'
    default_use_math = 'True'
    default_image_topic = '/quadruped/camera/rgb'
    default_camera_info_topic = '/quadruped/camera/camera_info'
    default_twist_joy_enable_button = "5"
    default_twist_joy_linear_x_button = "4"
    default_twist_joy_angular_yaw_button = "3"  

    # Create launch description
    ld = launch.LaunchDescription()

    # Declare launch arguments with proper defaults
    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'model_server_url',
            description='Model server URL',
            default_value=default_model_server_url,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'token',
            description='Access token for the model server',
            default_value=default_token,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'use_math',
            description='If set to true, use mathematical approach for reading gauges instead of neural network.',
            default_value=default_use_math,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'twist_joy_enable_button',
            description='',
            default_value=default_twist_joy_enable_button,
        )
    )


    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'twist_joy_linear_linear_x_button',
            description='',
            default_value=default_twist_joy_linear_x_button,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'twist_joy_angular_yaw_button',
            description='',
            default_value=default_twist_joy_angular_yaw_button,
        )
    )



    image_topic = ld.add_action(
        launch.actions.DeclareLaunchArgument('image_topic', default_value=default_image_topic)
    )

    camera_info_topic = ld.add_action(
        launch.actions.DeclareLaunchArgument('camera_info_topic', default_value=default_camera_info_topic)
    )


    model_server_url = LaunchConfiguration('model_server_url', default = default_model_server_url)
    token = LaunchConfiguration('token', default = default_token)
    use_math = LaunchConfiguration('use_math', default = default_use_math)
    image_topic = LaunchConfiguration('image_topic', default=default_image_topic)
    camera_info_topic = LaunchConfiguration('camera_info_topic', default=default_camera_info_topic)
    twist_joy_enable_button = LaunchConfiguration('twist_joy_enable_button', default=default_twist_joy_enable_button)
    twist_joy_linear_linear_x_button = LaunchConfiguration('twist_joy_linear_linear_x_button', default=default_twist_joy_linear_x_button)
    twist_joy_angular_yaw_button = LaunchConfiguration('twist_joy_angular_yaw_button', default=default_twist_joy_angular_yaw_button)


    rectify_node = ComposableNode(
        package='image_proc',
        plugin='image_proc::RectifyNode',
        name='rectify',
        namespace='apriltag',
        remappings=[('image', image_topic),
                    ('camera_info', camera_info_topic)],
    )

    apriltag_node = ComposableNode(
        package='apriltag_ros',
        plugin='AprilTagNode',
        name='apriltag',
        namespace='apriltag',
        parameters=[{
            'family': '36h11',
            'size': 0.0766,
        }],
        remappings=[
            ('image_rect', '/apriltag/image_rect'),
            ('camera_info', camera_info_topic),
            ('tf', '/tf')
        ],
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
        output='screen',
        on_exit=on_exit,
    )

    ld.add_action(apriltag_container)


    # Create LaunchConfigurations
   
    # QoS configuration file
    qos_config = os.path.join(package_share_dir, 'config', 'qos_config_lite.yaml')
    # Parameters configuration file
    param_config = os.path.join(package_share_dir, 'config', 'config_lite.yaml')


    # Add gauge_reader node
    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='gauge_reader_lite',
            name='gauge_reader',
            parameters=[
                qos_config,
                param_config,
                {
                    'model_server_url': model_server_url,
                    'token': token,
                    'use_math': use_math,
                }
            ],
            on_exit = on_exit
        )
    )

    ld.add_action(
        launch_ros.actions.Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_node',
            output='screen',
            parameters=[
                {'enable_button': twist_joy_enable_button},
                {'axis_angular.yaw': twist_joy_angular_yaw_button},
                {'axis_linear.x': twist_joy_linear_linear_x_button},
                {'scale_linear.x': 2.0},
                {'scale_angular.z': 2.0}
            ],
            remappings=[
                ('joy_vel', 'cmd_vel')
            ],
            on_exit = on_exit
        )
    )

    ld.add_action(
        launch_ros.actions.Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
    )

    return ld
