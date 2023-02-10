import rclpy
from gym import Env
from gym.spaces import Discrete, Dict, Box
import numpy as np
from hospital_robot_spawner.robot_controller import RobotController
import math
#from rcl_interfaces.srv import GetParameters

class HospitalBotEnv(RobotController, Env):
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
        self._target_location = np.array([1, 10], dtype=np.float32) # Training [1, 10]
        # If True, at each episode, the target location is set randomly, otherwise it is fixed
        self._randomize_target = False
        # If True, the observation space is normalized between [0,1]
        self._normalize_obs = True
        # If True, the action space is normalized between [-1,1]
        self._normalize_act = True
        # Chooses the reward method to use - simple reward, heuristic, adaptive heuristic (Checkout the method compute_reward)
        self._reward_method = "simple reward"
        # Initializes the maximal linear velocity used in actions
        self._max_linear_velocity = 1
        # Initializes the minimal linear velocity used in actions
        self._min_linear_velocity = 0.2
        # Initializes the angular velocity used in actions - This must always be symmetric (no need for max and min)
        self._angular_velocity = 1
        # Initializes the min distance from target for which the episode is concluded with success
        self._minimum_dist_from_target = 0.35
        # Initializes the min distance from an obstacle for which the episode is concluded without success
        self._minimum_dist_from_obstacles = 0.35
        # Attraction threshold and factor for adaptive heuristic
        self._attraction_threshold = 3
        self._attraction_factor = 1
        # Repulsion threshold and factor for adaptive heuristic
        self._repulsion_threshold = 1
        self._repulsion_factor = 0.1
        # Distance penalty factor
        self._distance_penalty_factor = 1

        # Initialize step count
        self._num_steps = 0

        # Debug prints on console
        self.get_logger().info("TARGET LOCATION: " + str(self._target_location))
        self.get_logger().info("AGENT LOCATION: " + str(self._agent_location))
        self.get_logger().info("MAX LINEAR VEL: " + str(self._max_linear_velocity))
        self.get_logger().info("MIN LINEAR VEL: " + str(self._min_linear_velocity))
        self.get_logger().info("ANGULAR VEL: " + str(self._angular_velocity))
        self.get_logger().info("MIN TARGET DIST: " + str(self._minimum_dist_from_target))
        self.get_logger().info("MIN OBSTACLE DIST: " + str(self._minimum_dist_from_obstacles))

        if self._normalize_act == True:
            ## Normalized Action space - It is a 2D continuous space - Linear velocity, Angular velocity
            self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        else:
            ## Action space - It is a 2D continuous space - Linear velocity, Angular velocity
            self.action_space = Box(low=np.array([self._min_linear_velocity, -self._angular_velocity]), high=np.array([self._max_linear_velocity, self._angular_velocity]), dtype=np.float32)
        
        if self._normalize_obs == True:
            ## Normalized State space - dictionary with: "Robot position", "Laser reads"
            self.observation_space = Dict(
                {
                    # Agent position can be anywhere inside the hospital, its limits are x=[-12;12] and y=[-35,21]
                    "agent": Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32),
                    # Laser reads are 61 and can range from 0.08 to 10
                    "laser": Box(low=-1, high=1, shape=(61,), dtype=np.float32),
                }
            )

        else:
            ## State space - dictionary with: "Robot position", "Laser reads"
            self.observation_space = Dict(
                {
                    # Agent position can be anywhere inside the hospital, its limits are x=[-12;12] and y=[-35,21]
                    #"agent": Box(low=np.array([-12, -35]), high=np.array([12, 21]), dtype=np.float32),
                    "agent": Box(low=np.array([0, -math.pi]), high=np.array([60, math.pi]), dtype=np.float32),
                    # Laser reads are 61 and can range from 0.08 to 10
                    "laser": Box(low=0, high=np.inf, shape=(61,), dtype=np.float32),
                }
            )

    def step(self, action):

        # Increase step number
        self._num_steps += 1

        # De-normalize the action to send the command to robot
        if self._normalize_act == True:
            action = self.denormalize_action(action)

        # Apply the action
        #self.get_logger().info("Action applied: " + str(action))
        self.send_velocity_command(action)

        # Spin the node until laser reads and agent location are updated - VERY IMPORTANT
        self._previous_agent_location = self._agent_location
        self._previous_laser_reads = self._laser_reads
        self.spin()

        # Compute the polar coordinates of the robot with respect to the target
        self.transform_coordinates()

        # Update robot location and laser reads
        observation = self._get_obs()
        #self.get_logger().info(str(observation["laser"]))
        # Update distance from target
        info = self._get_info()
        # Check if episode is terminated
        done = (info["distance"] < self._minimum_dist_from_target) or (any(info["laser"] < self._minimum_dist_from_obstacles))
        #self.get_logger().info("Done: " + str(done))

        # Compute Reward
        reward = self.compute_rewards(observation, info)

        return observation, reward, done, info

    def render(self):
        # Function to render env steps
        pass

    def reset(self, seed=None, options=None):
        #self.get_logger().info("Resetting the environment")

        # Reset the done reset variable
        self._done_reset_sim = False
        self._done_reset_env = False
        # Calls the reset simulation service
        self.call_reset_simulation_service()
        # Here we spin the node until the /reset_simulation service responds, otherwise we get random observations
        while self._done_reset_sim == False:
            rclpy.spin_once(self)
        self.call_reset_environment_service()
        # Here we spin the node until the /reset_environment service responds, otherwise we get random observations
        while self._done_reset_env == False:
            rclpy.spin_once(self)

        # Randomize target location
        if self._randomize_target == True:
            self.randomize_target_location()

        # Compute the initial observation
        self._previous_agent_location = self._agent_location
        self._previous_laser_reads = self._laser_reads
        self.spin()
        self.transform_coordinates()

        # Updates state and additional infos
        observation = self._get_obs()
        info = self._get_info()

        # Reset the number of steps
        self._num_steps = 0

        # Debug print
        #self.get_logger().info("Exiting reset function")
        
        return observation

    def _get_obs(self):
        # Returns the current state of the system
        obs = {"agent": self._polar_coordinates, "laser": self._laser_reads}
        # Normalize observations
        if self._normalize_obs == True:
            obs = self.normalize_observation(obs)
        #self.get_logger().info("Agent Location: " + str(self._agent_location))
        return obs

    def _get_info(self):
        # returns the distance from agent to target and laser reads
        return {
            "distance": math.dist(self._agent_location, self._target_location),
            "laser": self._laser_reads,
            "angle": self._polar_coordinates[1]
        }

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        while np.array_equal(self._laser_reads, self._previous_laser_reads) or np.array_equal(self._agent_location, self._previous_agent_location):
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

        ## Heuristic reward
        if self._reward_method == "heuristic":
            if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward
                reward = 0
                self.get_logger().info("TARGET REACHED")
                #self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))
            elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                reward = -1000
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                # Otherwise the episode continues
                reward = -1

        ## Heuristic with Adaptive Exploration Strategy
        elif self._reward_method == "adaptive heuristic":
            if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward minus the time it spent to reach it
                reward = 1000 - self._num_steps
                self.get_logger().info("TARGET REACHED")
                #self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))
            elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                reward = -10000
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                # Otherwise the episode continues
                ## Instant reward of the current state - euclidean distance
                instant_reward = -info["distance"] * self._distance_penalty_factor

                ## Estimated reward of the current state (attraction-repulsion rule)
                # Attraction factor - Activates when the agent is near the target
                if info["distance"] < self._attraction_threshold:
                    attraction_reward = self._attraction_factor / info["distance"]
                else:
                    attraction_reward = 0
                # Repulsion factor - Activates when the agent is near any obstacles
                contributions = [((-self._repulsion_factor / read**2)*((1/read) - (1/self._repulsion_threshold))) \
                     for read in info["laser"] if read<=self._repulsion_threshold]
                repulsion_reward = sum(contributions)
                # Compute final reward - Min between this and +1 to creating a higher reward than reaching the target
                reward = min(instant_reward + attraction_reward + repulsion_reward,1)

        ## Simple reward
        elif self._reward_method == "simple reward":
            if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward
                reward = 1
                self.get_logger().info("TARGET REACHED")
                #self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))
            elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                reward = -1
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                # Otherwise the episode continues
                reward = 0
            
        return reward

    def randomize_target_location(self):
        ## This method sets the target location at a semi-random position
        self._target_location = np.array([1, 10], dtype=np.float32) 
        self._target_location[0] += np.float32(np.random.rand(1)*6-3) # Random contr. on target x ranges in [-3,+3]
        self._target_location[1] += np.float32(np.random.rand(1)*4-1) # Random contr. on target x ranges in [-1,+3]
        #self.get_logger().info("TARGET LOCATION: " + str(self._target_location))

    def normalize_observation(self, observation):
        ## This method normalizes the observations taken from the robot in the range [0,1]
        # Distance from target can range from 0 to 60 but we divide by ten since most times the agent never goes further
        observation["agent"][0] = observation["agent"][0]/10
        # Angle from target can range from -pi to pi
        observation["agent"][1] = (observation["agent"][1] + math.pi)/(2*math.pi)
        # Laser reads range from 0 to 10
        observation["laser"] = observation["laser"]/10

        # Debug
        #self.get_logger().info("Agent: " + str(observation["agent"]))
        #self.get_logger().info("Laser: " + str(observation["laser"]))
        
        return observation

    def denormalize_action(self, norm_act):
        ## This method de-normalizes the action before sending it to the robot - The action is normalized between [-1,1]
        # Linear velocity can also not be symmetric
        action_linear = ((self._max_linear_velocity*(norm_act[0]+1)) + (self._min_linear_velocity*(1-norm_act[0])))/2
        # Angular velicity is symmetric
        action_angular = ((self._angular_velocity*(norm_act[1]+1)) + (-self._angular_velocity*(1-norm_act[1])))/2

        # Debug
        #self.get_logger().info("Linear velocity: " + str(action_linear))
        #self.get_logger().info("Angular velocity: " + str(action_angular))

        return np.array([action_linear, action_angular], dtype=np.float32)
    
    def close(self):
        ## Shuts down the node to avoid creating multiple nodes on re-creation of the env
        self.destroy_client(self.client_sim)
        self.destroy_client(self.client_env)
        self.destroy_publisher(self.action_pub)
        self.destroy_subscription(self.pose_sub)
        self.destroy_subscription(self.laser_sub)
        self.destroy_node()