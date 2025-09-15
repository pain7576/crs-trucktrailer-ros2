import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from custom_msg.msg import MyWorldAngle

class Carstatepublisher(Node):
    def __init__(self):
        super().__init__('car_state_pub')
        self.sub1 = self.create_subscription(PoseStamped,'BEN_CAR_WIFI/pose', self.car_pos_pub_callback,10)
        self.sub2 = self.create_subscription(MyWorldAngle,'angle',self.car_ori_callback,10)
        self.car_state = [0, 0]
        self.origin_my_world = [0.30958878993988037, 0.7059913873672485]

    def car_pos_world_2_my_world(self, x, y):
        car_state_x = (x - self.origin_my_world[0])*100
        car_state_y = (y - self.origin_my_world[1])*100

        return car_state_x, car_state_y

    def car_pos_pub_callback(self, msgRigidBodyPose):
        self.car_state[0], self.car_state[1] = self.car_pos_world_2_my_world(msgRigidBodyPose.pose.position.x,msgRigidBodyPose.pose.position.y)

    def car_ori_callback(self, msg1):
        angle =  msg1.angle
        self.get_logger().info(f'car state x:{self.car_state[0]}, y:{self.car_state[1]}, angle:{angle}°')

def main():
    rclpy.init()
    carstatepublisher = Carstatepublisher()
    rclpy.spin(carstatepublisher)
    carstatepublisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




