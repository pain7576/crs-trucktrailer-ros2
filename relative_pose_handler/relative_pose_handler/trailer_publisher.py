import rclpy
from rclpy.node import Node
from custom_msg.msg import CarState
from custom_msg.msg import TruckTrailerState
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import scipy.integrate as spi
import numpy as np
import math


class TrailerPublisher(Node):
    def __init__(self):
        super().__init__('trailer_publisher')

        # Subscribing to truck's state: [x, y, angle]
        self.sub = self.create_subscription(CarState, '/car_state', self.truck_state_callback, 10)

        # Publisher for trailer pose
        self.pub = self.create_publisher(TruckTrailerState, '/truck_trailer_pose', 10)

        # Constants
        self.L1 = 9  # Truck length (from rear axle to front axle)
        self.L2 = 10  # Trailer length (from its axle to the hitch point)
        self.hitch_offset = 0  # Offset from rear axle of truck to hitch point
        self.v1x = 0.0  # Forward speed of truck (cm/s), initialized to 0
        self.steering_angle = np.deg2rad(13)  # In radians (adjust as needed)

        # State is now separated for clarity and correctness
        self.truck_state = None  # [psi1, x1, y1] - Truck heading, x, y (rear axle)
        self.trailer_state = None  # [psi2, x2, y2] - Trailer heading, x, y (trailer axle)

        self.initialized = False
        self.simulation_time = 0  # simulation time
        self.last_truck_x = None
        self.last_truck_y = None
        self.last_timestamp = None

        # Configuration space for the environment
        self.min_map_x = -75
        self.min_map_y = -75
        self.max_map_x = 75
        self.max_map_y = 75

    def truck_state_callback(self, car_msg):
        current_time = self.get_clock().now()
        # The incoming message represents the truck's rear axle position and heading
        truck_x = round(car_msg.x)
        truck_y = round(car_msg.y)
        psi1 = np.deg2rad(car_msg.angle)

        if not self.initialized:
            # Initialize both truck and trailer states on the first message
            self.truck_state = np.array([psi1, truck_x, truck_y])

            # Trailer is initially aligned with truck, placed directly behind it
            x2 = truck_x - self.L2 * math.cos(psi1)
            y2 = truck_y - self.L2 * math.sin(psi1)
            psi2 = psi1
            self.trailer_state = np.array([psi2, x2, y2])
            self.initialized = True
            self.last_timestamp = current_time
            self.last_truck_x = truck_x
            self.last_truck_y = truck_y
        else:
            # Calculate actual dt from message timing
            dt = (current_time - self.last_timestamp).nanoseconds / 1e9

            if dt > 0.001:  # Only proceed if meaningful time has passed
                # Estimate velocity based on actual time elapsed
                distance = np.sqrt((truck_x - self.last_truck_x) ** 2 + (truck_y - self.last_truck_y) ** 2)
                self.v1x = distance / dt
                if self.v1x > 15:
                    self.v1x = 25

                # Update truck state
                self.truck_state = np.array([psi1, truck_x, truck_y])

                # Integrate trailer with actual dt (not fixed dt)
                if not np.isclose(self.v1x, 0.0):
                    sol = spi.solve_ivp(
                        fun=lambda t, y: self.kinematic_model(t, y),
                        t_span=[self.simulation_time, self.simulation_time + dt],  # Use actual dt
                        y0=self.trailer_state,
                        method='RK45'
                    )
                    self.trailer_state = sol.y[:, -1]

                self.simulation_time += dt

                # Publish and render immediately after each update
                self.publish_trailer_state()
                self.render()

            else:
                # If dt is too small, just update truck state without integrating
                self.truck_state = np.array([psi1, truck_x, truck_y])

        self.last_truck_x = truck_x
        self.last_truck_y = truck_y
        self.last_timestamp = current_time

    def publish_trailer_state(self):
        # Publish the combined state of the truck and trailer
        truck_trailer_state = TruckTrailerState()
        # Trailer state from the integrator
        truck_trailer_state.psi2 = self.trailer_state[0]
        truck_trailer_state.x2 = self.trailer_state[1]
        truck_trailer_state.y2 = self.trailer_state[2]
        # Truck state from the subscriber
        truck_trailer_state.psi1 = self.truck_state[0]
        truck_trailer_state.x1 = self.truck_state[1]
        truck_trailer_state.y1 = self.truck_state[2]
        self.pub.publish(truck_trailer_state)

    def kinematic_model(self, t, trailer_y):
        """
        Kinematic model of the TRAILER ONLY.
        It calculates the trailer's motion based on the truck's current state.
        """
        # Unpack current truck state (from the subscriber)
        psi_1, _, _ = self.truck_state
        # Unpack current trailer state (from the integrator)
        psi_2, _, _ = trailer_y

        # --- Calculate Trailer Motion ---
        # Hitch angle difference is critical for trailer kinematics
        hitch_angle = psi_1 - psi_2

        # Truck's angular rate (needed for trailer calculations)
        dpsi_1 = (self.v1x / self.L1) * np.tan(self.steering_angle)

        # Trailer's forward velocity
        v2x = self.v1x * np.cos(hitch_angle) + (self.hitch_offset * dpsi_1 * np.sin(hitch_angle))

        # Trailer's angular rate (how fast the trailer turns)
        dpsi_2 = (self.v1x / self.L2) * np.sin(hitch_angle) - (
                (self.hitch_offset / self.L2) * dpsi_1 * np.cos(hitch_angle))

        # Trailer's position derivatives (how fast it moves in x and y)
        dx2 = v2x * np.cos(psi_2)
        dy2 = v2x * np.sin(psi_2)

        # Return the derivatives for the trailer's state vector [dpsi_2, dx2, dy2]
        return np.array([dpsi_2, dx2, dy2])

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

        psi1, x1, y1 = self.truck_state
        psi2, x2, y2 = self.trailer_state

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
    node = TrailerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()