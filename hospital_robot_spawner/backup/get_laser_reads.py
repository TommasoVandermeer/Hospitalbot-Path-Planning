#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LaserSubscriber(Node):

    def __init__(self):
        super().__init__("laser_reads")
        self.get_logger().info("The Laser Subscriber has been successfully created")
        self.laser_sub = self.create_subscription(LaserScan, '/demo/laser/out', self.laser_callback, 10)

    def laser_callback(self, msg: LaserScan):
        self.get_logger().info(str(np.array(msg.ranges)))


def main(args=None):
    rclpy.init(args=args)
    node = LaserSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()