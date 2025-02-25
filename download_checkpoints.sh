#!/bin/bash

wget -P checkpoints https://huggingface.co/hcltech-robotics/gauge_detect/resolve/main/gauge_detect.pt gauge_detect.pt
wget -P checkpoints https://huggingface.co/hcltech-robotics/gauge_reader/resolve/main/gauge_net_ResidualSEBlock_with_boxes.pt gauge_net_ResidualSEBlock_with_boxes.pt