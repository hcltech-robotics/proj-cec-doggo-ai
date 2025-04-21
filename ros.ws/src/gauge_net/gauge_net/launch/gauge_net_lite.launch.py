import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
import launch_ros.actions


def generate_launch_description():
    package_name = 'gauge_net'

    # Get package share directory
    package_share_dir = get_package_share_directory(package_name)

    # Define default paths for model weights inside the installed package
    model_server_url = 'http://localhost:5000'
    token = 'doggodoggo'

    # Create launch description
    ld = launch.LaunchDescription()

    # Declare launch arguments with proper defaults
    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'model_server_url',
            description='Model server URL',
            default_value=model_server_url,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'token',
            description='Access token for the model server',
            default_value=token,
        )
    )

    # Create LaunchConfigurations
    model_server_url = LaunchConfiguration('model_server_url')
    token = LaunchConfiguration('token')

    # QoS configuration file
    qos_config = os.path.join(package_share_dir, 'config', 'qos_config_lite.yaml')
    # Parameters configuration file
    param_config = os.path.join(package_share_dir, 'config', 'config_lite.yaml')

    print(param_config)

    # Add gauge_reader node
    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='gauge_reader_lite',
            name='gauge_reader',
            parameters=[
                {
                    'model_server_url': model_server_url,
                    'token': token,
                },
                qos_config,
                param_config,
            ],
        )
    )

    return ld
