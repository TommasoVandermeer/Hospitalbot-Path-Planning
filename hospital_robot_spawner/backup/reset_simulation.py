#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from functools import partial

class ResetSimService(Node):

    def __init__(self):
        super().__init__("reset_sim")
        self.call_reset_simulation_service()

    def call_reset_simulation_service(self):
        client = self.create_client(Empty, "/reset_simulation")
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = Empty.Request()

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_reset_simulation))

    def callback_reset_simulation(self, future):
        try:
            response= future.result()
            self.get_logger().info("The Simulation has been successfully reset")
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

def main(args=None):
    rclpy.init(args=args)
    node = ResetSimService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
