import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import time

class GlobalParameterServer(Node):

    def __init__(self):
        super().__init__("global_parameters", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # Create service to request parameters from any node
        self.create_service(GetParameters, 'download_parameters', self.parameters_callback)

    def parameters_callback(self, request, response):
        response.values = [self.get_parameter(key).get_parameter_value() for key in request.names]
        self.get_logger().info('Incoming request\nParameters requested: ' + str(request.names))

        return response

def main(args=None):
    rclpy.init()
    node = GlobalParameterServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()