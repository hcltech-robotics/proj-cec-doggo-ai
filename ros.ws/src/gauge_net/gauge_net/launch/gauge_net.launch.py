import os

from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions


def generate_launch_description():
    # Create an empty launch description
    ld = launch.LaunchDescription()

    # Declare launch arguments
    gauge_detector_weights = launch.substitutions.LaunchConfiguration('gauge_detector_weights')
    gauge_reader_weights = launch.substitutions.LaunchConfiguration('gauge_reader_weights')
    image_topic = launch.substitutions.LaunchConfiguration('image_topic', default='/image')

    # QoS configuration file
    qos_config = os.path.join(
        get_package_share_directory('gauge_net'), 'config', 'qos_config.yaml'
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'gauge_detector_weights', description='Path to weights for gauge_detector'
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'gauge_reader_weights', description='Path to weights for gauge_reader'
        )
    )

    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'image_topic', description='Image topic to subscribe to'
        )
    )

    # Add gauge_reader node
    ld.add_action(
        launch_ros.actions.Node(
            package='gauge_net',
            executable='gauge_reader',
            name='gauge_reader',
            parameters=[
                {
                    'detector_model_file': gauge_detector_weights,
                    'reader_model_file': gauge_reader_weights,
                    'image_topic': image_topic,
                },
                qos_config,
            ],
        )
    )

    return ld
