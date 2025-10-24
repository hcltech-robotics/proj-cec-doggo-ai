#!/bin/bash
eval "$(conda shell.bash hook)"

conda create -n omniverse-5.1 -y python==3.11

conda activate omniverse-5.1

pip install -r requirements5.1.txt

