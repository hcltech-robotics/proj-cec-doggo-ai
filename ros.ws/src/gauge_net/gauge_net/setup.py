from setuptools import find_packages, setup

package_name = "gauge_net"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/gauge_net.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="istvan.fodor",
    maintainer_email="info@istvanfodor.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gauge_detector = gauge_net.gauge_detector:main",
            "gauge_reader = gauge_net.gauge_reader:main",
        ]
    },
)
