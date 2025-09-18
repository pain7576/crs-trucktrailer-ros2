import rclpy
from rclpy.node import Node
from custom_msg.msg import CarState
from custom_msg.msg import TruckTrailerState
from crs_msgs.msg import CarSteerState
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import numpy as np
import math
import sympy as sp
from sympy import symbols, Matrix, sin, cos, tan
from filterpy.kalman import ExtendedKalmanFilter as EKF


class RobotEKF(EKF):
    """
    Extended Kalman Filter for a Truck and Trailer System.

    State vector (x): [x1, y1, psi1, x2, y2, psi2]
        - (x1, y1): Truck's rear axle position
        - psi1: Truck's heading angle (radians)
        - (x2, y2): Trailer's axle position
        - psi2: Trailer's heading angle (radians)

    Measurement vector (z): [x1, y1, psi1]
        - Direct measurements of the truck's pose.

    Control vector (u): [steering_angle, v1x]
        - delta (steering_angle): Truck's steering angle (radians)
        - v1x: Truck's forward velocity
    """

    def __init__(self, dt, L1, L2, hitch_offset):
        # State vector: [x1, y1, psi1, x2, y2, psi2]
        super().__init__(dim_x=6, dim_z=3, dim_u=2)

        self.dt = dt
        self.L1 = L1
        self.L2 = L2
        self.hitch_offset = hitch_offset

        # --- Setup Symbolic Model ---
        self._setup_symbolic_model()

        # --- Noise Covariances ---
        # Measurement noise (R): uncertainty of our sensors (e.g., GPS, IMU)
        std_x, std_y, std_psi = 0.05, 0.05, np.deg2rad(0.1)
        self.R = np.diag([std_x ** 2, std_y ** 2, std_psi ** 2])

        # Process noise (Q): uncertainty in the kinematic model
        std_pos_proc = 0.01
        std_psi_proc = np.deg2rad(0.1)
        self.Q = np.diag([
            std_pos_proc ** 2, std_pos_proc ** 2, std_psi_proc ** 2,
            std_pos_proc ** 2, std_pos_proc ** 2, std_psi_proc ** 2
        ])

        # Control noise (M): uncertainty in control inputs
        std_steer = np.deg2rad(3.0)
        std_v = 2
        self.M = np.diag([std_steer ** 2, std_v ** 2])

        # Measurement function H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

    def _setup_symbolic_model(self):
        """
        Derives the state transition function and its Jacobians using sympy.
        """
        x1_s, y1_s, psi1_s, x2_s, y2_s, psi2_s = symbols('x1 y1 psi1 x2 y2 psi2')
        delta_s, v1x_s = symbols('delta v1x')
        state_s = Matrix([x1_s, y1_s, psi1_s, x2_s, y2_s, psi2_s])
        control_s = Matrix([delta_s, v1x_s])

        # Kinematic model
        hitch_s = psi1_s - psi2_s
        dpsi1_s = (v1x_s / self.L1) * tan(delta_s)
        v2x_s = v1x_s * cos(hitch_s) + self.hitch_offset * dpsi1_s * sin(hitch_s)
        dpsi2_s = (v1x_s / self.L2) * sin(hitch_s) - (self.hitch_offset / self.L2) * dpsi1_s * cos(hitch_s)
        dx1_s = v1x_s * cos(psi1_s)
        dy1_s = v1x_s * sin(psi1_s)
        dx2_s = v2x_s * cos(psi2_s)
        dy2_s = v2x_s * sin(psi2_s)

        state_dot_s = Matrix([dx1_s, dy1_s, dpsi1_s, dx2_s, dy2_s, dpsi2_s])

        # Jacobians
        F_j = state_dot_s.jacobian(state_s)
        G_j = state_dot_s.jacobian(control_s)

        # Create fast, callable functions
        self.state_dot_func = sp.lambdify((state_s, control_s), state_dot_s, 'numpy')
        self.F_jacobian_func = sp.lambdify((state_s, control_s), F_j, 'numpy')
        self.G_jacobian_func = sp.lambdify((state_s, control_s), G_j, 'numpy')

    def predict(self, u):
        """
        Predicts the next state and covariance using the control input u.
        """
        # --- Predict State (using Euler integration) ---
        state_dot = self.state_dot_func(self.x, u).flatten()
        self.x = self.x + state_dot * self.dt

        # Normalize truck heading (psi1)
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        # Normalize trailer heading (psi2)
        self.x[5] = np.arctan2(np.sin(self.x[5]), np.cos(self.x[5]))

        # --- Predict Covariance ---
        F = self.F_jacobian_func(self.x, u)
        G = self.G_jacobian_func(self.x, u)
        F_k = np.eye(self.dim_x) + F * self.dt
        self.P = F_k @ self.P @ F_k.T + G @ self.M @ G.T * self.dt ** 2 + self.Q

    def update(self, z):
        """
        Updates the state estimate with a new measurement z.
        """
        super().update(z, HJacobian=lambda x: self.H, Hx=lambda x: self.H @ x, R=self.R)


class EKFTrailerPublisher(Node):
    def __init__(self):
        super().__init__('ekf_trailer_publisher')

        # Subscriptions
        self.sub = self.create_subscription(CarState, '/car_state', self.truck_state_callback, 10)
        self.sub1 = self.create_subscription(CarSteerState, '/car_steer_state', self.steer_callback, 10)

        # Publisher
        self.pub = self.create_publisher(TruckTrailerState, '/truck_trailer_pose', 10)

        # Constants
        self.L1 = 9.0
        self.L2 = 10.0
        self.hitch_offset = 0.0

        # Control inputs
        self.v1x = 0.0
        self.steering_angle = 0.0

        # EKF Initialization
        # dt is initialized to a small number; it will be updated dynamically.
        self.ekf = RobotEKF(dt=0.01, L1=self.L1, L2=self.L2, hitch_offset=self.hitch_offset)

        # State tracking
        self.initialized = False
        self.last_truck_x = None
        self.last_truck_y = None
        self.last_timestamp = None

        # Environment configuration
        self.min_map_x, self.max_map_x = -75, 75
        self.min_map_y, self.max_map_y = -75, 75

    def truck_state_callback(self, car_msg):
        current_time = self.get_clock().now()
        truck_x = car_msg.x
        truck_y = car_msg.y
        psi1 = np.deg2rad(car_msg.angle)

        # Measurement vector from incoming message
        z = np.array([truck_x, truck_y, psi1])

        if not self.initialized:
            # Initialize EKF state on the first message
            x2 = truck_x - self.L2 * math.cos(psi1)
            y2 = truck_y - self.L2 * math.sin(psi1)
            psi2 = psi1
            self.ekf.x = np.array([truck_x, truck_y, psi1, x2, y2, psi2])

            # Initialize covariance with high uncertainty
            self.ekf.P = np.eye(6) * 500.0

            self.initialized = True
        else:
            dt = (current_time - self.last_timestamp).nanoseconds / 1e9
            if dt > 1e-4:  # Ensure a meaningful time step
                # Estimate velocity
                distance = np.sqrt((truck_x - self.last_truck_x) ** 2 + (truck_y - self.last_truck_y) ** 2)
                self.v1x = distance / dt

                # Update EKF's internal timestep
                self.ekf.dt = dt

                # --- EKF Cycle ---
                # 1. Predict the next state based on control inputs
                control_input = np.array([self.steering_angle, self.v1x])
                self.ekf.predict(u=control_input)

                # 2. Update the state with the new measurement
                self.ekf.update(z=z)

                # Publish the estimated state and render
                self.publish_trailer_state()
                self.render()

        # Update values for next iteration
        self.last_truck_x = truck_x
        self.last_truck_y = truck_y
        self.last_timestamp = current_time

    def steer_callback(self, msg):
        self.steering_angle = msg.steer_angle - np.deg2rad(2)

    def publish_trailer_state(self):
        if not self.initialized:
            return

        # Unpack estimated state from EKF
        x1, y1, psi1, x2, y2, psi2 = self.ekf.x

        msg = TruckTrailerState()
        msg.x1 = x1
        msg.y1 = y1
        msg.psi1 = psi1
        msg.x2 = x2
        msg.y2 = y2
        msg.psi2 = psi2
        self.pub.publish(msg)

    def plot_vehicle(self, ax, x, y, heading, length, width, label, color='blue', show_wheels=True, steering_angle=0.0):
        """Plot vehicle visualization."""
        dx = 0
        dy = -width / 2
        rect = Rectangle((dx, dy), length, width, linewidth=1.5, edgecolor='black',
                         facecolor=color, alpha=0.6)
        t = Affine2D().rotate_around(0, 0, heading).translate(x, y)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

        if show_wheels:
            rear_axle_pos = np.array([0, 0])
            front_axle_pos = np.array([length, 0])

            wheel_radius = width * 0.22
            wheel_offset_y = width / 2 * 0.9
            wheel_dir_len = wheel_radius * 2.0

            def transform_point(p):
                R = np.array([[np.cos(heading), -np.sin(heading)],
                              [np.sin(heading), np.cos(heading)]])
                return R @ p + np.array([x, y])

            # Rear wheels
            rear_left_wheel_center = transform_point(rear_axle_pos + np.array([0, wheel_offset_y]))
            rear_right_wheel_center = transform_point(rear_axle_pos + np.array([0, -wheel_offset_y]))

            for center in [rear_left_wheel_center, rear_right_wheel_center]:
                circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=1,
                                        zorder=5)
                ax.add_patch(circle_outline)

            for center in [rear_left_wheel_center, rear_right_wheel_center]:
                line_end = center + wheel_dir_len * np.array([np.cos(heading), np.sin(heading)])
                ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=1, zorder=6)

            if label.lower() == 'truck':
                front_left_wheel_center = transform_point(front_axle_pos + np.array([0, wheel_offset_y]))
                front_right_wheel_center = transform_point(front_axle_pos + np.array([0, -wheel_offset_y]))

                for center in [front_left_wheel_center, front_right_wheel_center]:
                    circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=1,
                                            zorder=5)
                    ax.add_patch(circle_outline)

                front_wheel_heading = heading + steering_angle
                for center in [front_left_wheel_center, front_right_wheel_center]:
                    line_end = center + wheel_dir_len * np.array(
                        [np.cos(front_wheel_heading), np.sin(front_wheel_heading)])
                    ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=1, zorder=7)

            rear_axle_line_start = transform_point(rear_axle_pos + np.array([0, -wheel_offset_y * 1.1]))
            rear_axle_line_end = transform_point(rear_axle_pos + np.array([0, wheel_offset_y * 1.1]))
            ax.plot([rear_axle_line_start[0], rear_axle_line_end[0]],
                    [rear_axle_line_start[1], rear_axle_line_end[1]], 'k-', linewidth=2.0, zorder=4)

            if label.lower() == 'truck':
                front_axle_line_start = transform_point(front_axle_pos + np.array([0, -wheel_offset_y * 1.1]))
                front_axle_line_end = transform_point(front_axle_pos + np.array([0, wheel_offset_y * 1.1]))
                ax.plot([front_axle_line_start[0], front_axle_line_end[0]],
                        [front_axle_line_start[1], front_axle_line_end[1]], 'k-', linewidth=2.0, zorder=4)

        arrow_color = 'blue' if label.lower() == 'trailer' else color
        arrow_length = width * 1.5
        arrow_dx = arrow_length * np.cos(heading)
        arrow_dy = arrow_length * np.sin(heading)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.4, head_length=0.6,
                 fc=arrow_color, ec=arrow_color, linewidth=2, zorder=10)

    def render(self):
        """Render the environment."""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()

        self.ax.clear()

        x1, y1, psi1, x2, y2, psi2 = self.ekf.x

        L1, W1 = self.L1, 2.0
        L2, W2 = self.L2, 2.0
        delta = self.steering_angle  # Use actual steering angle
        hitch_offset = 0.0

        # Draw vehicles
        self.plot_vehicle(self.ax, x1, y1, psi1, L1, W1, label='Truck', color='blue', show_wheels=True,
                          steering_angle=delta)
        self.plot_vehicle(self.ax, x2, y2, psi2, L2, W2, label='Trailer', color='green', show_wheels=True)

        # Draw hitch connection
        hitch_x = x1 - hitch_offset * np.cos(psi1)
        hitch_y = y1 - hitch_offset * np.sin(psi1)
        trailer_front_x = x2 + L2 * np.cos(psi2)
        trailer_front_y = y2 + L2 * np.sin(psi2)
        self.ax.plot([hitch_x, trailer_front_x], [hitch_y, trailer_front_y], 'r-', linewidth=2, label='Hitch Link')

        self.ax.set_xlim(self.min_map_x, self.max_map_x)
        self.ax.set_ylim(self.min_map_y, self.max_map_y)
        self.ax.set_aspect('equal')

        # Calculate hitch distance for debugging
        hitch_distance = np.sqrt((hitch_x - trailer_front_x) ** 2 + (hitch_y - trailer_front_y) ** 2)

        info_text = (
            f"Truck: x={x1:.1f}, y={y1:.1f}, ψ={np.rad2deg(psi1):.0f}°, δ={np.rad2deg(delta):.0f}°\n"
            f"Trailer: x={x2:.1f}, y={y2:.1f}, ψ={np.rad2deg(psi2):.0f}°\n"
            f"Velocity: {self.v1x:.1f} cm/s, Hitch gap: {hitch_distance:.2f} cm"
        )
        self.ax.text(2.5, 21.0, info_text, fontsize=6, fontweight='bold', va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = EKFTrailerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()