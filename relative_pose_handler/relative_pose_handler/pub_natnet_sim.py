import rclpy
import numpy as np
from rclpy.node import Node
from custom_msg.msg import MyWorldAngle
from custom_msg.msg import WorldOrientation

class Natnet_simulator(Node):
    def __init__(self):
        super().__init__('natnet_publisher')
        self.publisher_ = self.create_publisher(WorldOrientation,'world_quaternion', 10)
        time_period = 0.5
        self.timer = self.create_timer(time_period, self.timer_callback)

    def random_quaternion(self):
        u1, u2, u3 = np.random.rand(3)  # Uniform random numbers in [0, 1)

        q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

        return np.array([q1, q2, q3, q4])

    def timer_callback(self):
        msg = WorldOrientation()
        rand_q = self.random_quaternion()
        msg.x = rand_q[0]
        msg.y = rand_q[1]
        msg.z = rand_q[2]
        msg.w = rand_q[3]
        self.publisher_.publish(msg)
        self.get_logger().info(str(msg))

def main():
    rclpy.init()
    natnet_simulator = Natnet_simulator()
    rclpy.spin(natnet_simulator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()



