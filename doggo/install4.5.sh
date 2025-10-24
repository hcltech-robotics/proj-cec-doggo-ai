#!/bin/bash
eval "$(conda shell.bash hook)"

conda create -n omniverse-4.5 -y python==3.10

conda activate omniverse-4.5

pip install -r requirements4.5.txt

