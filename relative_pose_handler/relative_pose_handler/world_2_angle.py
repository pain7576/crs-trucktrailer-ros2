import rclpy
import numpy as np
import math
from rclpy.node import Node
from custom_msg.msg import MyWorldAngle
from custom_msg.msg import WorldOrientation


class World2angle(Node):
    def __init__(self):
        super().__init__('quaternion2angle')
        self.subscription = self.create_subscription(WorldOrientation,'world_quaternion',self.quaternion2angle_callback,10)
        self.pubangle = self.create_publisher(MyWorldAngle,'angle',10)
        self.q1 = [0.30943127229359507,  0.9061490616051597, 0.07920361817209619, -0.27725972074447486]
        self.q2 = [0.0, 0.0, 0.0, 0.0]

    def quaternion_to_euler(self, q):
        w, x, y, z = q

        # Roll (X-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (Z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def angle_finder(self, q1 ,q2):
        r1, p1, y1 = self.quaternion_to_euler(q1)
        r2, p2, y2 = self.quaternion_to_euler(q2)

        # Calculate difference and convert to degrees
        d_roll = math.degrees(r2 - r1)
        d_pitch = math.degrees(p2 - p1)
        d_yaw = math.degrees(y2 - y1)

        return d_roll, d_pitch, d_yaw

    def quaternion2angle_callback(self, msg):
        self.q2[0] = msg.x
        self.q2[1] = msg.y
        self.q2[2] = msg.z
        self.q2[3] = msg.w

        diffx, diffy, diffz = self.angle_finder(self.q1,self.q2)
        self.publish_angle(diffx)

        self.get_logger().info(f'Angle differences - X: {diffx:.4f}°, Y: {diffy:.4f}°, Z: {diffz:.4f}°')

    def publish_angle(self, diffx):
        msg1 = MyWorldAngle()
        msg1.angle = diffx
        self.pubangle.publish(msg1)


def main():
    rclpy.init()
    world2angle = World2angle()
    rclpy.spin(world2angle)
    world2angle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




