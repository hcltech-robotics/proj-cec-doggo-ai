from setuptools import find_packages, setup
import glob
import os

package_name = 'artie_audio'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/artie_audio_stack.launch.py']),
        ('share/' + package_name + '/config', glob.glob('config/*')),
        ('share/' + package_name + '/models', glob.glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mic_streamer = artie_audio.mic_streamer:main',
            'kws_node = artie_audio.kws_node:main',
            'asr_node = artie_audio.asr:main'
        ],
    },
)
