from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import os
from ament_index_python.packages import get_package_share_directory
#from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class RobotController(Node):
    """
    This class defines all the methods to:
        - Publish actions to the agent (move the robot)
        - Subscribe to sensors of the agent (get laser scans and robot position)
        - Reset the simulation

    Topics list:
        - /demo/cmd_vel : linear and angular velocity of the robot
        - /demo/odom : odometry readings of the chassis of the robot
        - /demo/laser/out : laser readings
    
    Services used:
        - /reset_simulation : resets the gazebo simulation
        - /get_model_list: returns the list of models present in gazebo

    Services not used:
        - /delete_entity : unspawns the robot from the simulation
        - /spawn_entity : spawns the robot in the simulation in a semi-random position

    Gazebo topic:
        - /gazebo/world/pose/modify : instantly modifies the pose and orientation of any model
        Es: [gz topic -p /gazebo/world/pose/modify -m "name: 'HospitalBot', position: {x: 1, y: 10, z: 0}, orientation: {x: 0, y: 0, z: 1, w: 0}"]
    """
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has just been created")

        # Action publisher
        self.action_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        # Position subscriber
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 1)
        # Laser subscriber
        self.laser_sub = self.create_subscription(LaserScan, '/demo/laser/out', self.laser_callback, 1)
        # Reset simulation client
        self.client_sim = self.create_client(Empty, "/reset_simulation")
        # Reset environment client
        self.client_env = self.create_client(Empty, "/reset_environment")
        
        # Get the directory of the sdf of the robot
        self._pkg_dir = os.path.join(
            get_package_share_directory("hospital_robot_spawner"), "models",
        "pioneer3at", "model.sdf")

        # Initialize attributes - This will be immediately re-written when the simulation starts
        self._agent_location = np.array([np.float32(1),np.float32(16)])
        self._laser_reads = np.array([np.float32(10)] * 61)
    
    # Method to send the velocity command to the robot
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    # Method that saves the position of the robot each time the topic /demo/odom receives a new message
    def pose_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x,-12,12)), np.float32(np.clip(msg.pose.pose.position.y,-35,21))])
        self._agent_orientation = 2* math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        #self.get_logger().info("Agent position: " + str(self._agent_location))
        #self.get_logger().info("Agent orientation: " + str(math.degrees(self._agent_orientation)))

    # Method that saves the laser reads each time the topic /demo/laser/out receives a new message
    def laser_callback(self, msg: LaserScan):
        self._laser_reads = np.array(msg.ranges)
        # Converts inf values to 20
        self._laser_reads[self._laser_reads == np.inf] = np.float32(20)
        #self.get_logger().info("Min Laser Read: " + str(min(self._laser_reads)))

    # Method to reset the simulation (calls the service /reset_simulation)
    def call_reset_simulation_service(self):
        while not self.client_sim.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = Empty.Request()

        future = self.client_sim.call_async(request)
        future.add_done_callback(partial(self.callback_reset_simulation))

    # Method that elaborates the future obtained by callig the /reset_simulation service
    def callback_reset_simulation(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Simulation has been successfully reset")
            self._done_reset_sim = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

        # Method to reset the simulation (calls the service /reset_environment)
    def call_reset_environment_service(self):
        while not self.client_env.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = Empty.Request()

        future = self.client_env.call_async(request)
        future.add_done_callback(partial(self.callback_reset_environment))

    # Method that elaborates the future obtained by callig the /reset_environment service
    def callback_reset_environment(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
            self._done_reset_env = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    # Method to unspawn the robot from the simulation - NOT USED
    def call_delete_entity_service(self):
        client = self.create_client(DeleteEntity, '/delete_entity')
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = DeleteEntity.Request()
        request.name = 'HospitalBot'

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_delete_entity))

    # Method that elaborates the future obtained by callig the /delete_entity service  - NOT USED
    def callback_delete_entity(self, future):
        try:
            response= future.result()
            self.get_logger().info("The Robot has been successfully UN-spawned")
            self._done_delete = True
        except Exception as e:
            self.get_logger().error("DeleteEntity Service call failed: %r" % (e,))

    # Method to spawn the robot inside the simulation at any given position  - NOT USED
    def call_spawn_entity_service(self):
        client = self.create_client(SpawnEntity, '/spawn_entity')
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SpawnEntity.Request()
        request.name = 'HospitalBot'
        request.xml = open(self._pkg_dir, 'r').read()
        request.robot_namespace = 'demo'
        # Here we set the position - Add random floats to x,y coordinates to make the training more generalizable
        request.initial_pose.position.x = float(1) + float(np.random.rand(1)*2-1) # + random float [-1 , 1]
        request.initial_pose.position.y = float(16) + float(np.random.rand(1) - 0.5)# + random float [-0,5 , 0,5]
        request.initial_pose.position.z = float(0)
        # Here we set the orientation - Add random float to the angle to make the training more generalizable
        desired_angle = float(math.radians(-90) + math.radians(np.random.rand(1)*60-30)) # + random float [-30, 30]
        request.initial_pose.orientation.z = float(math.sin(desired_angle/2))
        request.initial_pose.orientation.w = float(math.cos(desired_angle/2))

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_spawn_entity))

    # Method that elaborates the future obtained by callig the /spawn_entity service  - NOT USED
    def callback_spawn_entity(self, future):
        try:
            response= future.result()
            self.get_logger().info("The Robot has been successfully spawned")
            self._done_spawn = True
        except Exception as e:
            self.get_logger().error("SpawnEntity Service call failed: %r" % (e,))