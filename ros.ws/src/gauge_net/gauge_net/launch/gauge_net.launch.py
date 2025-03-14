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
    default_gauge_detector_weights = os.path.join(package_share_dir, 'models', 'gauge_detect.pt')
    default_gauge_reader_weights = os.path.join(package_share_dir, 'models', 'gauge_reader.pt')

    # Create launch description
    ld = launch.LaunchDescription()

    # Declare launch arguments with proper defaults
    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'gauge_detector_weights',
            description='Path to weights for gauge_detector',
            default_value=default_gauge_detector_weights,
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'gauge_reader_weights',
            description='Path to weights for gauge_reader',
            default_value=default_gauge_reader_weights,
        )
    )

    # Create LaunchConfigurations
    gauge_detector_weights = LaunchConfiguration('gauge_detector_weights')
    gauge_reader_weights = LaunchConfiguration('gauge_reader_weights')

    # QoS configuration file
    qos_config = os.path.join(package_share_dir, 'config', 'qos_config.yaml')
    # Parameters configuration file
    param_config = os.path.join(package_share_dir, 'config', 'config.yaml')

    # Add gauge_reader node
    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='gauge_reader',
            name='gauge_reader',
            parameters=[
                {
                    'detector_model_file': gauge_detector_weights,
                    'reader_model_file': gauge_reader_weights,
                },
                qos_config,
                param_config,
            ],
        )
    )

    return ld
