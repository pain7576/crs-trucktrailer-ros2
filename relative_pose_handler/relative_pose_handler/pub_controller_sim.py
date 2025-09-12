import rclpy
import numpy as np
import math
from rclpy.node import Node
from crs_msgs.msg import CarInput

class Controller_simulator(Node):
    def __init__(self):
        super().__init__('controller_publisher')
        self.publisher1 = self.create_publisher(CarInput,'control_input',10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg4 = CarInput()
        msg4.torque = math.nan
        msg4.velocity = 0.2
        msg4.steer = 0.22
        msg4.steer_override = False

        self.publisher1.publish(msg4)
        self.get_logger().info(str(msg4))

def main():
    rclpy.init()
    controller_simulator = Controller_simulator()
    rclpy.spin(controller_simulator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
