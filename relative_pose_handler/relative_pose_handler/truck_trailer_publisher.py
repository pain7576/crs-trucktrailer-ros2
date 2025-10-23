import rclpy
from rclpy.node import Node
from custom_msg.msg import CarState
from custom_msg.msg import TruckTrailerState
from custom_msg.msg import Shutdown
from custom_msg.msg import TrailerState
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import numpy as np
import math

class TruckTrailerReal(Node):
    def __init__(self):
        super().__init__('TruckTrailerReal')

        self.sub = self.create_subscription(CarState, '/car_state', self.truck_state_callback, 10)
        self.sub2 = self.create_subscription(TrailerState, 'trailer_state', self.trailer_state_callback, 10)
        self.shutdown_subscriber = self.create_subscription(Shutdown, '/shutdown_signal', self.shutdown_callback, 10)

        self.truck_x = 0
        self.truck_y = 0
        self.psi1 = 0
        self.trailer_x = 0
        self.trailer_y= 0
        self.psi2 = 0

        self.L1 = 9.0
        self.L2 = 7
        self.steering_angle = 0
        self.hitch_offset = 2.5

        # Environment configuration
        self.min_map_x, self.max_map_x = -75, 75
        self.min_map_y, self.max_map_y = -75, 75

        # Publisher
        self.pub = self.create_publisher(TruckTrailerState, '/truck_trailer_pose', 10)

    def truck_state_callback(self, car_msg):
        self.truck_x = car_msg.x
        self.truck_y = car_msg.y
        self.psi1 = np.deg2rad(car_msg.angle)
        self.publish_trailer_state()
        self.render()


    def trailer_state_callback(self, trailer_msg):
        self.trailer_x = trailer_msg.x
        self.trailer_y = trailer_msg.y
        self.psi2 = np.deg2rad(trailer_msg.angle)

    def shutdown_callback(self, msg):
        self.get_logger().info('Received shutdown signal. Shutting down EKF Trailer Publisher.')
        self.destroy_node()
        rclpy.shutdown()

    def publish_trailer_state(self):

        msg = TruckTrailerState()
        msg.x1 = self.truck_x
        msg.y1 = self.truck_y
        msg.psi1 = self.psi1
        msg.x2 = float(self.trailer_x)
        msg.y2 = float(self.trailer_y)
        msg.psi2 = float(self.psi2)
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

        psi1, psi2, x1, y1, x2, y2 = self.psi1, self.psi2, self.truck_x, self.truck_y, self.trailer_x, self.trailer_y

        L1, W1 = self.L1, 2.0
        L2, W2 = self.L2, 2.0
        delta = self.steering_angle
        hitch_offset = self.hitch_offset

        # Draw vehicles
        self.plot_vehicle(self.ax, x1, y1, psi1, L1, W1, label='Truck', color='blue', show_wheels=True,
                          steering_angle=delta)
        self.plot_vehicle(self.ax, x2, y2, psi2, L2, W2, label='Trailer', color='green', show_wheels=True)

        # Draw hitch connection
        hitch_x = x1 - hitch_offset * np.cos(psi1)
        hitch_y = y1 - hitch_offset * np.sin(psi1)
        self.ax.scatter(hitch_x, hitch_y, color='yellow', s=4, label='Hitch Point')
        self.ax.plot([hitch_x, x1], [hitch_y, y1], 'k-', linewidth=0.5, label='Hitch Arm')

        self.ax.set_xlim(self.min_map_x, self.max_map_x)
        self.ax.set_ylim(self.min_map_y, self.max_map_y)
        self.ax.set_aspect('equal')


        info_text = (
            f"Truck: x={x1:.1f}, y={y1:.1f}, ψ={np.rad2deg(psi1):.0f}°, δ={np.rad2deg(delta):.0f}°\n"
            f"Trailer: x={x2:.1f}, y={y2:.1f}, ψ={np.rad2deg(psi2):.0f}°\n"

        )
        self.ax.text(2.5, 21.0, info_text, fontsize=6, fontweight='bold', va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    rclpy.init()
    node = TruckTrailerReal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()