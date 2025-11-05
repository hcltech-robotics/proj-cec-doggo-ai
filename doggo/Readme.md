# Quadruped Simulation with Isaac Sim and ROS2

This repository provides an **Isaac Sim–based quadruped simulation** that integrates with ROS2 (Humble or Jazzy) for full sensor publishing and object detection pipelines.  
The simulation can spawn different quadruped robots (Spot or Go2) into various environments and broadcast camera, IMU, LiDAR, and odometry data to ROS2 topics, enabling downstream nodes to perform object detection and gauge reading.

---

## 🧩 Features

- ✅ Supports both **ROS2 Humble** and **ROS2 Jazzy**
- 🦾 Quadruped support:
  - **Go2** (Unitree)
  - **Spot** (Boston Dynamics)
- 🌍 Environment choices:
  - `default` – simple flat world
  - `warehouse` – Isaac built-in warehouse
  - `jetty` – external USD scene with a dock and gauge model
  - `office` – interior scene with a gauge object and colliders
- 📷 ROS2 sensor bridge for:
  - RGB camera (`/quadruped/camera/rgb`, `/quadruped/camera/camera_info`)
  - LiDAR (`/scan`, `/point_cloud`) - By default Lidar is off
  - IMU (`/imu`)
  - Odometry (`/odom`, `/tf`)
- 🤖 Receives `/cmd_vel` twist commands from external ROS2 nodes

---


## ⚙️ Installation

### 1️⃣ Choose Your Isaac Sim Version

Two install scripts are provided:

#### **Isaac Sim 4.5**
```bash
bash install4.5.sh
```

#### **Isaac Sim 5.1**

```bash
bash install5.1.sh
```


Both scripts create and configure a Conda environment with the appropriate dependency file. 

The conda environments are called `omniverse-4.5` and `omniverse-5.1` respectively.

## 🚀 Running the Simulation

To launch the simulation, use one of the provided runner scripts (humble.sh or jazzy.sh) which activate the correct ROS2 environment and launch Isaac Sim.

### Run under ROS2 Humble

```bash
./humble.sh python quadruped.py --env jetty --quadruped spot
```

### Run under ROS2 Jazzy
```bash
./jazzy.sh python quadruped.py --env office --quadruped go2 --lidar
```


## 🏗️ Command-Line Parameters

| Flag          | Type   | Default   | Choices                                   | Description                                     |
| ------------- | ------ | --------- | ----------------------------------------- | ----------------------------------------------- |
| `--env`       | string | `default` | `default`, `warehouse`, `jetty`, `office` | Environment USD scene to load                   |
| `--quadruped` | string | `go2`     | `go2`, `spot`                             | Quadruped robot model                           |
| `--lidar`     | flag   | *off*     | —                                         | Enable LiDAR sensor and ROS2 bridge publishing |


## 📡 ROS2 Integration

The simulation automatically enables the following Isaac Sim ROS2 extensions for ROS2 bridging with the following 
topics:

| Topic                           | Message Type              | Description       |
| ------------------------------- | ------------------------- | ----------------- |
| `/odom`                         | `nav_msgs/Odometry`       | Robot odometry    |
| `/imu`                          | `sensor_msgs/Imu`         | IMU data          |
| `/scan`                         | `sensor_msgs/LaserScan`   | 2D LiDAR scan     |
| `/point_cloud`                  | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `/quadruped/camera/rgb`         | `sensor_msgs/Image`       | RGB camera image  |
| `/quadruped/camera/camera_info` | `sensor_msgs/CameraInfo`  | Camera intrinsics |
| `/tf`                           | `tf2_msgs/TFMessage`      | Transform tree    |

## ROS2 Endpoints

To run the additional ROS2 capabilities for gauge detection run the following:

1. ROS2 endpoints: build the (ros.ws)[../ros.ws] folder with colcon and run `ros2 launch gauge_net gauge_net_lite.launch.py use_math:=True`
1. Model endpoint: from the project root, install the requirements.txt file in a virtual environment, activate, run `download_checkpoints.sh` then run the following: `python model_server.py --token doggodoggo --detector-model checkpoints/gauge_detect.pth --reader-model checkpoints/gauge_reader.pt`

## Foxglove integration

To view gauge image, detections, etc in Foxglove, run the following:

1. Source the ros.ws install folder: `source install/setup.sh`
2. Install and run the foxglove bridge: `ros2 launch foxglove_bridge foxglove_bridge_launch.xml`
3. Open the localhost connection
4. Open the `Gauge Reading Dashboard.json` dashboard
