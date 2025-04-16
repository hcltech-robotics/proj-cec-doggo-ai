

1. Go to the root of the project and run `./download_checkpoints.sh`
1. Run `git submodule update --init --recursive`
1. Go to the `ros.ws/src/go2_ros2_sdk` folder, run this: `docker build . -f docker/Dockerfile -t go2_ros_sdk:latest`
1. Go to the `web` folder and run this: `docker build . -t proj-cec-doggo-web:latest -f docker/webprod/Dockerfile`
1. Go to the parent of this folder
1. Run the following: `docker build . -f docker/Dockerfile -t robot-base:latest`
1. Find the Robot's IP address
1. Run the following: `ROBOT_IP="<robot't IP>" docker compose up ` 