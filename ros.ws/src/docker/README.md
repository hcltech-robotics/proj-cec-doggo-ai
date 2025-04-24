
## Running a dockerized version of the lite endpoints

1. Go to the root of the project and run `./download_checkpoints.sh`
1. Run `git submodule update --init --recursive`
1. Go to the `web` folder and run this: `docker build . -t proj-cec-doggo-web:latest -f docker/webprod/Dockerfile`
1. Go to the parent of this folder
1. Run the following: `docker build . -f docker/Dockerfile -t robot-base:latest`
1. Find the Robot's IP address
1. Run the model server on a host powerful enough to host Pytorch models.
    1. `python3 model_server.py --token <randomtoken> --detector-model checkpoints/gauge_detect.pth --reader-model checkpoints/gauge_reader.pt` 
1. Run the following: `ROBOT_IP="<robot't IP>" MODEL_SERVER_URL='http://<model server host and port>' && export TOKEN=<same random token as above> docker compose -f docker-compose-lite.yml up ` 