#!/bin/bash
# replace eth0 with your actual interface name (check with `ip link`)
sudo ip link set eth0 up
sudo ip addr add 192.168.123.222/24 dev eth0
