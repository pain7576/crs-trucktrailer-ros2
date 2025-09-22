import rclpy
from rclpy.node import Node
from custom_msg.msg import CarState
from custom_msg.msg import Shutdown
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import math
import numpy as np


class Car_state_publisher(Node):
    def __init__(self):
        super().__init__('car_state_publisher')
        self.carmocap = self.create_subscription(PoseStamped, 'BEN_CAR_WIFI/pose', self.latest_value_callback, 10)
        self.shutdown_subscriber = self.create_subscription(Shutdown, '/shutdown_signal', self.shutdown_callback, 10)
        self.carpub = self.create_publisher(CarState, 'car_state',10)
        self.throttle = self.create_timer(0.001, self.car_state_callback)
        self.q1 = [0.6530918478965759, -0.015541106462478638, -0.020415853708982468, 0.7568438649177551]
        self.q2 = [0, 0, 0, 0]
        self.origin_my_world = [0.30958878993988037, 0.7059913873672485]
        self.latest_x = 0
        self.latest_y = 0
        self.q_x = 0
        self.q_y = 0
        self.q_z = 0
        self.q_w = 0




    def latest_value_callback(self, msgRigidBodyPose):
        self.latest_x = (msgRigidBodyPose.pose.position.x)
        self.latest_y = (msgRigidBodyPose.pose.position.y)
        self.q_x =  msgRigidBodyPose.pose.orientation.x
        self.q_y = msgRigidBodyPose.pose.orientation.y
        self.q_z = msgRigidBodyPose.pose.orientation.z
        self.q_w = msgRigidBodyPose.pose.orientation.w

    def shutdown_callback(self, msg):
        self.get_logger().info('Received shutdown signal. Shutting down Car State Publisher.')
        self.destroy_node()
        rclpy.shutdown()

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

    def quaternion2angles(self):
        self.q2[0] = self.q_w
        self.q2[1] = self.q_x
        self.q2[2] = self.q_y
        self.q2[3] = self.q_z

        diffx, diffy, diffz = self.angle_finder(self.q1,self.q2)

        return diffx, diffy, diffz

    def car_pos_world_2_my_world(self, x, y):
        car_state_x = (x - self.origin_my_world[0])*100
        car_state_y = (y - self.origin_my_world[1])*100

        return car_state_x, car_state_y

    def car_state_callback(self):

        car_msg = CarState()
        angle_x, angle_y, angle_z = self.quaternion2angles()
        car_x, car_y = self.car_pos_world_2_my_world(self.latest_x , self.latest_y)

        car_msg.angle = angle_z
        car_msg.x = car_x
        car_msg.y = car_y - 5

        self.get_logger().info(f'car state x:{car_x:.2f} y:{car_y:.2f} angle:{angle_z:.2f}')

        self.carpub.publish(car_msg)

def main():
    rclpy.init()
    car_state_publisher = Car_state_publisher()
    rclpy.spin(car_state_publisher)
    car_state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()











