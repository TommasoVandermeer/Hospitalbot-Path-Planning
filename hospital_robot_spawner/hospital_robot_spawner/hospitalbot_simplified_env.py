import rclpy
from gym import Env
from gym.spaces import Discrete, Dict, Box
import numpy as np
from hospital_robot_spawner.robot_controller import RobotController
import math
#from rcl_interfaces.srv import GetParameters

class HospitalBotSimpleEnv(RobotController, Env):
    """
    This class defines the RL environment. Here are defined:
        - Action space
        - State space
        - Target location
    
    The methods available are:
        - step: makes a step of the current episode
        - reset: resets the simulation environment when an episode is finished
        - close: terminates the environment

    This class inherits from both RobotController and Env.
    Env is a standard class of Gymnasium library which defines the basic needs of an RL environment.
    
    RobotController is a ROS2 Node used to control the agent. It includes the following attributes:
        - _agent_location: current position of the robot
        - _laser_reads: current scan of the LIDAR
    
    And the following methods (only the ones usefull here):
        - send_velocity_command: imposes a velocity to the agent (the robot)
        - call_reset_simulation_service: resets the simulation
        - call_reset_environment_service: resets the robot position to a semi-random point
    """
    def __init__(self):
        
        # Initialize the Robot Controller Node
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        # ENVIRONMENT PARAMETERS
        self.robot_name = 'HospitalBot'
        # Initializes the Target location
        self._target_location = np.array([1, 12], dtype=np.float32) # Training [1, 12]
        # If True, at each episode, the target location is set randomly, otherwise it is fixed
        self._randomize_target = False
        # If True, the target will appear on the simulation - SET FALSE FOR TRAINING (slows down the training)
        self._visualize_target = True
        # Initializes the linear velocity used in actions
        self._linear_velocity = 1
        # Initializes the angular velocity used in actions
        self._angular_velocity = 1
        # Initializes the min distance from target for which the episode is concluded with success
        self._minimum_dist_from_target = 0.3
        # Initializes the radius from the target where the robot can navigate
        self._target_range = 3.5

        # Initialize step count
        self._num_steps = 0

        # Debug prints on console
        self.get_logger().info("TARGET LOCATION: " + str(self._target_location))
        self.get_logger().info("AGENT LOCATION: " + str(self._agent_location))
        self.get_logger().info("LINEAR VEL: " + str(self._linear_velocity))
        self.get_logger().info("ANGULAR VEL: " + str(self._angular_velocity))
        self.get_logger().info("MIN TARGET DIST: " + str(self._minimum_dist_from_target))
        self.get_logger().info("TARGET RANGE: " + str(self._target_range))

        # Warning for training
        if self._visualize_target == True:
            self.get_logger().info("WARNING! TARGET VISUALIZATION IS ACTIVATED, SET IT FALSE FOR TRAINING")

        # Action space - It is a 2D continuous space - Linear velocity, Angular velocity
        self.action_space = Box(low=np.array([0, -self._angular_velocity]), high=np.array([self._linear_velocity, self._angular_velocity]), dtype=np.float32)

        # State space - Polar cordinates of the target in the robot coordinates system
        # Agent position can be anywhere inside the hospital, its limits are x=[-12;12] and y=[-35,21]
        self.observation_space = Box(low=np.array([-100, -math.pi]), high=np.array([100, math.pi]), dtype=np.float32)

    def step(self, action):

        # Increase step number
        self._num_steps += 1

        # Apply the action
        #self.get_logger().info("Action applied: " + str(action))
        self.send_velocity_command(action)

        # Spin the node until laser reads and agent location are updated - VERY IMPORTANT
        self.spin()

        # Compute the polar coordinates of the robot with respect to the target
        self.transform_coordinates()

        # Update robot location and laser reads
        observation = self._get_obs()
        #self.get_logger().info(str(observation["laser"]))
        # Update distance from target
        info = self._get_info()
        # Check if episode is terminated
        done = (info["distance"] < self._minimum_dist_from_target) or (info["distance"] > self._target_range)
        #self.get_logger().info("Done: " + str(done))

        # Compute Reward
        reward = self.compute_rewards(observation, info)

        return observation, reward, done, info

    def render(self):
        # Function to render env steps
        pass

    def reset(self, seed=None, options=None):
        #self.get_logger().info("Resetting the environment")

        angle = float(-90)
        orientation_z = float(math.sin(angle/2))
        orientation_w = float(math.cos(angle/2))

        pose2d = np.array([1, 15, orientation_z, orientation_w], dtype=np.float32)

        # Reset the done reset variable
        self._done_set_rob_state = False
        # Call the set robot position service
        self.call_set_robot_state_service(pose2d)
        # Here we spin the node until the /set_entity_state service responds, otherwise we get random observations
        while self._done_set_rob_state == False:
            rclpy.spin_once(self)

        # Randomize target location+
        if self._randomize_target == True:
            self.randomize_target_location()

        # Here we set the new target position for visualization
        if self._visualize_target == True:
            self.call_set_target_state_service(self._target_location)

        # Compute the initial observation
        self.spin()
        self.transform_coordinates()

        # Updates state and additional infos
        observation = self._get_obs()

        # Reset the number of steps
        self._num_steps = 0

        # Debug print
        #self.get_logger().info("Exiting reset function")
        
        return observation

    def _get_obs(self):
        # Returns the current state of the system
        #self.get_logger().info("Agent Location: " + str(self._agent_location))
        return self._polar_coordinates

    def _get_info(self):
        # returns the distance from agent to target
        return {
            "distance": math.dist(self._agent_location, self._target_location),
            "angle": self._polar_coordinates[1]
        }

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose == False) or (self._done_laser == False):
            rclpy.spin_once(self)

    def transform_coordinates(self):
        # This function takes as input: robot position, robot orientation and target position
        # To compute the polar coordinates of the robot with respect to target

        # Radius
        self._radius = math.dist(self._agent_location, self._target_location)

        # Target X coordinate expressed in the robot cartesian system
        self._robot_target_x = math.cos(-self._agent_orientation)* \
            (self._target_location[0]-self._agent_location[0]) - \
                math.sin(-self._agent_orientation)*(self._target_location[1]-self._agent_location[1])
        
        # Target Y coordinate expressed in the robot cartesian system
        self._robot_target_y = math.sin(-self._agent_orientation)* \
            (self._target_location[0]-self._agent_location[0]) + \
                math.cos(-self._agent_orientation)*(self._target_location[1]-self._agent_location[1])

        # Angle between the robot and the target (clockwise positive)
        self._theta = math.atan2(self._robot_target_y, self._robot_target_x)

        # Here we initialize the final variable which will be passed to the Gym environment
        self._polar_coordinates = np.array([self._radius,self._theta], dtype=np.float32)

        # Debug prints for new cartesian systems transformation (gazebo -> robot)
        #self.get_logger().info("Theta: " + str(math.degrees(self._theta)) + "\n" + "Radius: " + str(self._radius))
        #self.get_logger().info("Distance from target: " + str(math.sqrt(self._robot_target_x**2 + self._robot_target_y**2)))
        #self.get_logger().info("Xt: " + str(self._robot_target_x) + " - Yt: " + str(self._robot_target_y))
        #self.get_logger().info("Polar coordinates: " + str(self._polar_coordinates))

    def compute_rewards(self, observation, info):
        # This method computes the reward of the step
        if (info["distance"] < self._minimum_dist_from_target):
            # If the agent reached the target it gets a positive reward
            reward = 1
            self.get_logger().info("TARGET REACHED")
        else:
            reward = 0
            
        return reward

    def randomize_target_location(self):
        ## This method sets the target location at a semi-random position
        self._target_location = np.array([1, 10], dtype=np.float32) 
        self._target_location[0] += np.float32(np.random.rand(1)*6-3)
        self._target_location[1] += np.float32(np.random.rand(1)*4-1)
        #self.get_logger().info("TARGET LOCATION: " + str(self._target_location))

    def close(self):
        ## Shuts down the node to avoid creating multiple nodes on re-creation of the env
        self.destroy_client(self.client_sim)
        self.destroy_client(self.client_env)
        self.destroy_publisher(self.action_pub)
        self.destroy_subscription(self.pose_sub)
        self.destroy_subscription(self.laser_sub)
        self.destroy_node()