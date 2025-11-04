from setuptools import find_packages, setup
import os

package_name = 'artie_chatgpt'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', [
            'resource/info.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='tar.gabor14@gmail.com',
    description='ROS2 node integrating OpenAI function calling for Artie robot',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'llm_node = artie_chatgpt.chatgpt_node:main'
        ],
    },
)
