#!/bin/bash

echo "Starting model download and setup..."

# Create the checkpoints directory if it doesn't exist
echo "Creating 'checkpoints' directory if needed..."
mkdir -p checkpoints 

# Download the models and overwrite if necessary
echo "Downloading gauge_detect.pt..."
wget -O checkpoints/gauge_detect.pt https://huggingface.co/hcltech-robotics/gauge_detect/resolve/main/gauge_detect3_hcl.pt

echo "Downloading gauge_reader.pt..."
wget -O checkpoints/gauge_reader.pt https://huggingface.co/hcltech-robotics/gauge_reader/resolve/main/gauge_net3v4_hcl.pt

# Ensure the target directory exists
echo "Ensuring target directory 'ros.ws/src/gauge_net/gauge_net/models' exists..."
mkdir -p ros.ws/src/gauge_net/gauge_net/models

# Copy the models to the ROS workspace and overwrite if necessary
echo "Copying gauge_detect.pt to ROS workspace..."
cp -f checkpoints/gauge_detect.pt ros.ws/src/gauge_net/gauge_net/models/

echo "Copying gauge_reader.pt to ROS workspace..."
cp -f checkpoints/gauge_reader.pt ros.ws/src/gauge_net/gauge_net/models/

echo "Model setup completed successfully!"
