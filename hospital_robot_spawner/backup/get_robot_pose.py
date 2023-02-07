#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class PoseSubscriber(Node):

    def __init__(self):
        super().__init__("bot_position")
        self.get_logger().info("The Pose Subscriber has been successfully created")
        self.laser_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 10)

    def pose_callback(self, msg: Odometry):
        #self.get_logger().info('x: ' + str(msg.pose.pose.position.x) + ' y: ' + str(msg.pose.pose.position.y))
        self.get_logger().info(str(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])))

def main(args=None):
    rclpy.init(args=args)
    node = PoseSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()