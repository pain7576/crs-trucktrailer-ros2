# ROS 2 Autonomous Truck-Trailer Maneuvering with DDPG

## Project Overview

This is a ROS 2 system for autonomous truck-trailer maneuvering using Deep Deterministic Policy Gradient (DDPG) reinforcement learning. The system integrates motion capture data, Extended Kalman Filter (EKF) state estimation, and neural network-based control.

![Replay 1](assets\video.gif)

## System Architecture

The project consists of three main ROS 2 packages:

-   **`relative_pose_handler`**: Contains the core control, state estimation, and visualization nodes.
-   **`wifi_com`**: Handles hardware communication with the physical vehicle via UDP.
-   **`custom_msg`**: Defines custom ROS 2 message types used across the system.

## Prerequisites

### System Requirements:

-   Ubuntu 22.04 (recommended)
-   ROS 2 Humble or later
-   Python 3.8+

### Python Dependencies:

```bash
pip install torch numpy filterpy sympy matplotlib
```

### ROS 2 Dependencies:

-   `rclpy`
-   `geometry_msgs`
-   `sensor_msgs`
-   `crs_msgs` (external package)

## Installation

1.  **Create a ROS 2 workspace:**

    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    ```

2.  **Clone the repository:**

    ```bash
    git clone <repository-url> crs-trucktrailer-ros2
    cd ~/ros2_ws
    ```

3.  **Install dependencies:**

    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

4.  **Build the workspace:**

    ```bash
    colcon build
    source install/setup.bash
    ```

## Package Structure

The `relative_pose_handler` package provides 7 executable nodes:

| Executable      | Purpose                                         |
| --------------- | ----------------------------------------------- |
| `pose`          | Car state publisher from motion capture data    |
| `sys`           | EKF-based trailer state estimator               |
| `control_sim`   | DDPG reinforcement learning controller          |
| `render`        | Combined truck-trailer visualization            |
| `natnet_sim`    | Motion capture simulation                       |
| `angle_finder`  | Quaternion to Euler conversion                  |
| `car_pose`      | Coordinate frame transformation                 |

## Running the System

1.  **Start WiFi Communication Node:**

    This node handles UDP communication with the physical car hardware on port 20211.

    ```bash
    ros2 launch wifi_com wifi_com.launch.py
    ```

2.  **Launch State Publishers:**

    The `pose` node processes motion capture data and publishes vehicle state topics. The `sys` node runs the Extended Kalman Filter to estimate the complete truck-trailer state.

    ```bash
    # Terminal 1: Car state publisher
    ros2 run relative_pose_handler pose

    # Terminal 2: EKF trailer estimator
    ros2 run relative_pose_handler sys
    ```

3.  **Start DDPG Controller:**

    The controller loads pre-trained DDPG models and publishes steering commands.

    ```bash
    ros2 run relative_pose_handler control_sim
    ```

4.  **(Optional) Visualization:**

    This provides a real-time matplotlib visualization of the truck-trailer system.

    ```bash
    ros2 run relative_pose_handler render
    ```

## Configuration

### System Parameters:

-   **Truck wheelbase:** 9.0 cm
-   **Trailer length:** 7.0 cm
-   **Hitch offset:** 2.5 cm
-   **Workspace:** [-75, 75] cm in both X and Y
-   **Goal position:** (0, -30) cm with 90° orientation

### Control Parameters:

-   **Constant reverse torque:** -0.085 (~0.25 m/s)
-   **Steering range:** ±15°
-   **Control frequency:** 1000 Hz

## Topic Interface

Key topics used in the system:

-   `/BEN_CAR_WIFI/pose`: Raw truck pose from the motion capture system.
-   `/trailer1/pose`: Raw trailer pose from the motion capture system.
-   `/car_state`: Processed truck state.
-   `/truck_trailer_pose`: Fused truck-trailer state from the EKF.
-   `control_input`: DDPG steering commands sent to the vehicle.
-   `/shutdown_signal`: System-wide termination signal.

## Shutdown

The system automatically shuts down when the trailer reaches the goal position (y < -30 cm). All nodes listen to the `/shutdown_signal` topic for graceful termination.

## Notes

-   Pre-trained DDPG models must be present in the `tmp/ddpg/` directory before running the controller.
-   The motion capture system must be configured to track rigid bodies named "BEN_CAR_WIFI" and "trailer1".
-   WiFi communication requires a proper network configuration to reach the physical car hardware.
-   The EKF uses SymPy for symbolic Jacobian computation, which may cause a delay on the first run.
