import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
import launch_ros.actions


def generate_launch_description():
    package_name = 'artie_audio'

    # Get package share directory
    package_share_dir = get_package_share_directory(package_name)

    # Define default paths for model weights inside the installed package
    default_voice_activation_model_path = os.path.join(package_share_dir, 'models', 'hey-artie.tflite')

    # Create launch description
    ld = launch.LaunchDescription()

    # Declare launch arguments with proper defaults
    ld.add_action(
        launch.actions.DeclareLaunchArgument(
            'voice_activation_model_path',
            description='Path to model for voice activation',
            default_value=default_voice_activation_model_path,
        )
    )

    # Create LaunchConfigurations
    voice_activation_model_path = LaunchConfiguration('voice_activation_model_path')

    # Parameters configuration file
    kws_config = os.path.join(package_share_dir, 'config', 'kws_config.yaml')
    mic_streamer_config = os.path.join(package_share_dir, 'config', 'mic_streamer_config.yaml')
    asr_config = os.path.join(package_share_dir, 'config', 'asr_config.yaml')
    tts_config = os.path.join(package_share_dir, 'config', 'tts_config.yaml')

    # Add KWS node
    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='mic_streamer',
            name='mic_streamer',
            parameters=[
                mic_streamer_config,
            ],
        )
    )

    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='kws_node',
            name='kws_node',
            parameters=[
                {
                    'model_path': voice_activation_model_path,
                },
                kws_config,
            ],
        )
    )

    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='asr_node',
            name='asr_node',
            parameters=[
                asr_config,
            ],
        )
    )

    ld.add_action(
        launch_ros.actions.Node(
            package=package_name,
            executable='tts_node',
            name='tts_node',
            parameters=[
                tts_config,
            ],
        )
    )
    return ld
