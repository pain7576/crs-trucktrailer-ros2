import rclpy
import numpy as np
import math
from rclpy.node import Node
from custom_msg.msg import MyWorldAngle
from custom_msg.msg import WorldOrientation
from geometry_msgs.msg import PoseStamped


class World2angle(Node):
    def __init__(self):
        super().__init__('quaternion2angle')
        self.subscription = self.create_subscription(PoseStamped,'BEN_CAR_WIFI/pose',self.quaternion2angle_callback,10)
        self.pubangle = self.create_publisher(MyWorldAngle,'angle',10)
        self.q1 = [0.6530918478965759, -0.015541106462478638, -0.020415853708982468, 0.7568438649177551] #[w,x,y,z]
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

    def angle_finder(self, q1, q2):
        r1, p1, y1 = self.quaternion_to_euler(q1)
        r2, p2, y2 = self.quaternion_to_euler(q2)

        # Calculate difference and convert to degrees
        d_roll = math.degrees(r2 - r1)
        d_pitch = math.degrees(p2 - p1)
        d_yaw = math.degrees(y2 - y1)

        # --- SOLUTION: NORMALIZE THE YAW ANGLE ---
        # Wrap the angle to the range [-180, 180]
        if d_yaw > 180:
            d_yaw -= 360
        elif d_yaw < -180:
            d_yaw += 360
        # You can do the same for d_roll and d_pitch if needed

        return d_roll, d_pitch, d_yaw

    def quaternion2angle_callback(self, msgRigidBodyPose):
        self.q2[0] = msgRigidBodyPose.pose.orientation.w
        self.q2[1] = msgRigidBodyPose.pose.orientation.x
        self.q2[2] = msgRigidBodyPose.pose.orientation.y
        self.q2[3] = msgRigidBodyPose.pose.orientation.z

        diffx, diffy, diffz = self.angle_finder(self.q1,self.q2)
        self.publish_angle(diffz)

        self.get_logger().info(f'Angle differences - X: {diffx:.4f}°, Y: {diffy:.4f}°, Z: {diffz:.4f}°')

    def publish_angle(self, angle):
        msg1 = MyWorldAngle()
        msg1.angle = angle
        self.pubangle.publish(msg1)


def main():
    rclpy.init()
    world2angle = World2angle()
    rclpy.spin(world2angle)
    world2angle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




