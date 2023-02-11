import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
import subprocess
import numpy as np
import math

class ResetNode(Node):
    """
    This node is used to reset the simulation when the agent finishes its episode. It does one thing:
    - Calls /gazebo/world/pose/modify topic (Gazebo topic) to set a semi-random initial position
    """
    def __init__(self):
        super().__init__('reset_node')
        self.get_logger().info("The reset node has just been created")
        self.robot_name = "HospitalBot"

        # Set base robot position
        self.robot_initial_x = 1 # For simplified env is 1
        self.robot_initial_y = 16 # For simplified env is 14.5
        self.robot_initial_orientation = -90

        self.reset_env_srv = self.create_service(Empty, 'reset_environment', self.reset_environment_callback, )
        self.reset_target_srv = self.create_service(SetModelState, 'reset_target', self.reset_target_callback)

    def reset_environment_callback(self, request, response):
        # This method has to be outside the gym env because otherwise it makes the simulation crash
        
        #self.get_logger().info("Incoming request")

        ## Now, using a python subprocess, we set the semi-random initial position
        # Set the semi-random initial position
        position_x = float(self.robot_initial_x) + float(np.random.rand(1)*2-1) # Random contribution [-1,1]
        position_y = float(self.robot_initial_y) + float(np.random.rand(1) - 0.5) # Random contribution [-0.5,0.5]
        desired_angle = float(math.radians(self.robot_initial_orientation) + math.radians(np.random.rand(1)*60-30)) # Random contribution [-30,+30]
        orientation_z = float(math.sin(desired_angle/2))
        orientation_w = float(math.cos(desired_angle/2))
        position = f'{{x: {str(position_x)}, y: {str(position_y)}, z: 0}}'
        orientation = f'{{x: 0, y: 0, z: {str(orientation_z)}, w: {str(orientation_w)}}}'
        msg = f"name: '{self.robot_name}', position: {position}, orientation: {orientation}"
        # Since it is easier I created a subprocess (terminal) where I publish on a gazebo topic
        # Otherwise it would have been necessary to create a Gazebo plugin to publish on the topic from Ros2
        #self.get_logger().info("MSG: " + f"{msg}")
        try: 
            subprocess.run(["gz","topic","-p","/gazebo/world/pose/modify", "-m", f"{msg}"], close_fds=True, timeout=10)
        except Exception as e:
            self.get_logger().error("Subprocess run failed: %r" % (e,))

        return response

    def reset_target_callback(self, request: SetModelState.Request, response: SetModelState.Response):
        # Also this method has to be outside the gym env
        name = request.model_state.model_name
        position_x = request.model_state.pose.position.x
        position_y = request.model_state.pose.position.y
        position = f'{{x: {str(position_x)}, y: {str(position_y)}, z: 0.01}}'
        orientation = f'{{x: 0, y: 0, z: 0, w: 0}}'
        msg = f"name: '{name}', position: {position}, orientation: {orientation}"
        # Cast the subprocess to call the gazebo service
        #self.get_logger().info("MSG: " + f"{msg}")

        subprocess.run(["gz","topic","-p","/gazebo/world/pose/modify", "-m", f"{msg}"], close_fds=True)
        
        return response

def main(args=None):
    rclpy.init()
    node = ResetNode()

    rclpy.spin(node)

    node.get_logger().info("Done! Shutting down reset node.")
    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()