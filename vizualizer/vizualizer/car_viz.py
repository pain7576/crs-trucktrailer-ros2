import rclpy
import matplotlib.pyplot as plt
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class CarVisualizer(Node):
    def __init__(self):
        super().__init__('car_viz')
        self.carvizsub = self.create_subscription(PoseStamped, 'BEN_CAR_WIFI/pose', self.latest_value_callback, 10)
        self.latest_x = 0
        self.latest_y = 0

        self.timer = self.create_timer(0.5, self.viz_callback)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo')

        self.ax.set_title('live position of car')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_xlim(-50, 10)
        self.ax.set_ylim(60, 120)
        plt.show(block=False)

    def latest_value_callback(self, msgRigidBodyPose):
        self.latest_x = (msgRigidBodyPose.pose.position.x)*100
        self.latest_y = (msgRigidBodyPose.pose.position.y)*100

    def viz_callback(self):
        self.update_plot(round(self.latest_x, 3),round(self.latest_y, 3))
        self.get_logger().info(f'x:{self.latest_x},y:{self.latest_y}')


    def update_plot(self, x, y):
        self.line.set_xdata([x])
        self.line.set_ydata([y])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def main():
    rclpy.init()
    carVisualizer = CarVisualizer()
    rclpy.spin(carVisualizer)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

