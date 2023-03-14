import rclpy
from gym import Env
from gym.spaces import Dict, Box
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
        - _agent_location: current position of the robot (in gazebo's coordinate system)
        - _laser_reads: current scan of the LIDAR
    
    And the following methods (only the ones usefull here):
        - send_velocity_command: imposes a velocity to the agent (the robot)
        - call_reset_simulation_service: resets the simulation
        - call_reset_robot_service: resets the robot position to desired position
        - call_reset_target_service: resets the target position to desired position
    """
    def __init__(self):
        
        # Initialize the Robot Controller Node
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        # ENVIRONMENT PARAMETERS
        self.robot_name = 'HospitalBot'
        # Initializes the Target location (x,y) - effective only for randomization level 0 and 1 (see below)
        self._target_location = np.array([5, 0], dtype=np.float32) # Default is [1, 10]
        # Initializes the starting agent location for each episode (x,y,angle) - effective only for randomization level 0 and 2 (see below)
        self._initial_agent_location = np.array([1, 16, -90], dtype=np.float32) # Default is [1, 16, -90]
        # Defines the level of randomization of the env, the more you randomize the more the model will be generalizable (no overfitting)
        # 0: no randomization
        # 1: semi-randomize only robot's initial position
        # 2: semi-randomize only target location
        # 3: semi-randomize both robot position and target location
        # 4: semi-randomize both robot position and target location with obstacles (Door test)
        # 5: max randomization (both target and robot are reset in many locations at each episode)
        # 6: path planning mode (the robot has to reach several targets to complete the path)
        self._randomize_env_level = 0
        # If True, the observation space is normalized between [0,1] (except distance which is between [0,6], see below)
        self._normalize_obs = True
        # If True, the action space is normalized between [-1,1]
        self._normalize_act = True
        # If True, the target will appear on the simulation - SET FALSE FOR TRAINING (slows down the training)
        self._visualize_target = True
        # 0: simple reward, 1: heuristic, 2: adaptive heuristic (Checkout the method compute_reward)
        self._reward_method = 0
        # Initializes the maximal linear velocity used in actions
        self._max_linear_velocity = 1
        # Initializes the minimal linear velocity used in actions
        self._min_linear_velocity = 0
        # Initializes the angular velocity used in actions - This must always be symmetric (no need for max and min)
        self._angular_velocity = 1
        # Initializes the min distance from target for which the episode is concluded with success
        self._minimum_dist_from_target = 0.35
        # Initializes the min distance from an obstacle for which the episode is concluded without success
        self._minimum_dist_from_obstacles = 0.27
        
        ## Adaptive heuristic parameters
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
        self.get_logger().info("AGENT LOCATION: " + str(self._initial_agent_location))
        self.get_logger().info("MAX LINEAR VEL: " + str(self._max_linear_velocity))
        self.get_logger().info("MIN LINEAR VEL: " + str(self._min_linear_velocity))
        self.get_logger().info("ANGULAR VEL: " + str(self._angular_velocity))
        self.get_logger().info("MIN TARGET DIST: " + str(self._minimum_dist_from_target))
        self.get_logger().info("MIN OBSTACLE DIST: " + str(self._minimum_dist_from_obstacles))

        # Warning for training
        if self._visualize_target == True:
            self.get_logger().info("WARNING! TARGET VISUALIZATION IS ACTIVATED, SET IT FALSE FOR TRAINING")

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
                    # Maximum distance can be 60 but in 95% of cases it never goes over 10
                    "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
                    # Laser reads are 61 and can range from 0.08 to 10
                    "laser": Box(low=0, high=1, shape=(61,), dtype=np.float32),
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
        
        # This variable defines all the possible locations where the robot can spawn with a self._randomize_env_level of 5
        # [x, y, angle, x_lowerbound, x_upperbound, y_lowerbound, y_upperbound, angle_lowerbound, angle_upperbound]
        # x, y and angle defines the center of the location and the orientation, the bounds define the limits from the center in which the robot will spawn
        self.robot_locations = [[1, 16, -90, -1, 1, -0.5, 0.5, -30, 30],
                          [11, 13, 180, -1, 1, -0.25, 0.25, -30, 30],
                          [11.5, 5.7, 180,-0.2, 0.2, -0.1, 0.1, 0, 30],
                          [7.7, -8, 90, -0.5, 0.5, -1.5, 1.5, -30, 30],
                          [-2.3, -30.5, 90, -0.2, 0.5, -0.5, 1.5, -20, 45],
                          [-7.7, -30.8, 90, -0.2, 0.2, -0.5, 1, -20, 20],
                          [-2.1, -24.6, 180, -1, 1, -0.3, 0.3, -45, 30],
                          [-5, -6.6, 90, -0.5, 0.5, -1, 1, -30, 30],
                          [-5, 4, -90, -0.5, 0.5, -1, 1, -30, 30],
                          [-3.6, 10.9, 180, -1, 1.5, -1, 1, -15, 45],
                          [-1.6, -8.5, 0, -1, 1, -0.5, 0.5, -30, 30],
                          [5, -6, 90, -0.8, 0, -1, 1, -45, 45],
                          [2.8, -14.5, -90, 0, 0, -0.2, 0, -15, 15]]
        
        # This variable defines all the possible locations where the target can spawn with a self._randomize_env_level of 5
        # Obviously these locations are strictly associated with the locations where the robot spawns
        # [x, y, x_lowerbound, x_upperbound, y_lowerbound, y_upperbound]
        # x and y defines the center of the location, the bounds define the limits from the center in which the target will spawn
        self.target_locations = [[1, 10, -3, 3, -1, 3],
                          [6.7, 12, -0.1, 0.1, -0.5, 2],
                          [8, 4.8, -1.5, 1.5, -0.1, 0.1],
                          [10.8, -2.1, -1, 1, -0.1, 0.1],
                          [4.3, -27.6, -2, 0.5, -0.5, 0.5],
                          [-10.5, -26.3, -0.2, 1, -1, 1],
                          [-5, -21, -1, 1, -2.5, 0.5],
                          [-3.1, -3.5, -0.1, 0.1, -1, 1],
                          [0, 2, -3, 0, -1, 1],
                          [-7, 10.3, -0.5, 0.5, -0.5, 0.5],
                          [3.3, -8.6, -0.5, 0.5, -0.5, 0.5],
                          [3, -3.5, 0, 0, -1, 1],
                          [1.5, -19, -0.1, 0.5, -1, 1]]
        
        # This variable defines all the possible waypoints that can build the path
        # [x, y, x_lowerbound, x_upperbound, y_lowerbound, y_upperbound]
        # x and y defines the center of the location, the bounds define the limits from the center in which the target will spawn
        self.waypoints_locations = [[2, 10, -2, 2, -1, 1],
                                    [4, 5, -0.5, 0.5, -0.5, 0.5],
                                    [5, 0, -0.7, 1, -1, 1],
                                    [5, -5, -0.7, 1, -1, 1],
                                    [3, -8.5, -1, 1, -0.5, 0.5],
                                    [-3, -8.5, -1, 1, -0.5, 0.5],
                                    [-5, -13, -0.5, 0.5, -1, 1],
                                    [-5, -17.5, -0.5, 0.5, -1, 1],
                                    [-4.5, -25, 0, 1, 0, 0],
                                    [-9, -26, -1, 1, -0.5, 0.5],
                                    [-7.5, -31, 0, 0, -0.5, 0.5],
                                    [-7.5, -34, -1, 1, -0.5, 0.5],
                                    [-2, -33.5, -1, 1, -0.5, 0.5],
                                    [2, -33.5, -1, 1, -0.5, 0.5],
                                    [6, -33.5, -1, 1, -0.5, 0.5],
                                    [10, -33.5, -1, 1, -0.5, 0.5]]

        # This variable takes into account which point in the path the robot has to reach
        self._which_waypoint = 0

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
        # Update infos
        info = self._get_info()

        # Compute Reward
        reward = self.compute_rewards(info)

        # Check if episode is terminated
        if (self._randomize_env_level <= 5):
            # RANDOM LEVEL 1,2,3,4,5: Once the robot reaches the target or crashes into an obstacle the episode finishes
            done = (info["distance"] < self._minimum_dist_from_target) or (any(info["laser"] < self._minimum_dist_from_obstacles))
        else:
            # RANDOM LEVEL 6: Here the robot has to reach the target several times following a pre-determined path
            if (self._which_waypoint == len(self.waypoints_locations)-1):
                done = (info["distance"] < self._minimum_dist_from_target) or (any(info["laser"] < self._minimum_dist_from_obstacles))
            else:
                done = (any(info["laser"] < self._minimum_dist_from_obstacles))
                # Update waypoint
                if (info["distance"] < self._minimum_dist_from_target):
                    # Increase the variable to account for next waypoint
                    self._which_waypoint += 1
                    # Set the new waypoint
                    self.randomize_target_location()
                    # Here we set the new waypoint position for visualization
                    if self._visualize_target == True:
                        self.call_set_target_state_service(self._target_location)

        #self.get_logger().info("Done: " + str(done))

        return observation, reward, done, info

    def render(self):
        # Function to render env steps
        # Our env is always rendered if you start gazebo_world.launch.py
        # If you don't want to see graphical output launch headless_world.launch.py, it is also much faster for training (>20 times faster)
        pass

    def reset(self, seed=None, options=None):
        #self.get_logger().info("Resetting the environment")

        # Get the new pose of the robot
        pose2d = self.randomize_robot_location()
        
        # Reset the done reset variable
        self._done_set_rob_state = False
        # Call the set robot position service
        self.call_set_robot_state_service(pose2d)
        # Here we spin the node until the /set_entity_state service responds, otherwise we get random observations
        while self._done_set_rob_state == False:
            rclpy.spin_once(self)
        
        # Set waypoint back to 0 - ONLY VALID FOR RANDOM LEVEL 6
        self._which_waypoint = 0

        if (self._randomize_env_level >= 2):
        # Randomize target location
            self.randomize_target_location()

        # Here we set the new target position for visualization
        if self._visualize_target == True:
            self.call_set_target_state_service(self._target_location)

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
            "angle": self._theta
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

    def randomize_target_location(self):
        ## This method randomizes target position based on self._randomize_env_level (2, 3, 4 or 5)
        # Random Level 0 and 1 never enter here - Target location is still

        # RANDOM LEVEL 2 and 3 - Single location randomization
        if (self._randomize_env_level == 2) or (self._randomize_env_level == 3):
            self._target_location = np.array([1, 10], dtype=np.float32) # Base position [1,10]
            self._target_location[0] += np.float32(np.random.rand(1)*6-3) # Random contr. on target x ranges in [-3,+3]
            self._target_location[1] += np.float32(np.random.rand(1)*4-1) # Random contr. on target y ranges in [-1,+3]
            #self.get_logger().info("TARGET LOCATION: " + str(self._target_location))

        # RANDOM LEVEL 4 - Door test
        if (self._randomize_env_level == 4):
            self._target_location = np.array([6.45, 16.55], dtype=np.float32) # Base position
            self._target_location[0] += np.float32(np.random.rand(1)*3-0.1) # Random contr. on target x
            self._target_location[1] += np.float32(np.random.rand(1)*2.1-2) # Random contr. on target y
        
        # RANDOM LEVEL 5 - Max randomization
        if (self._randomize_env_level == 5):
            # The location was already chosen in randomize_robot_location
            self._target_location = np.array([self.target_locations[self._location][0], self.target_locations[self._location][1]], dtype=np.float32) # Base position
            self._target_location[0] += np.float32(np.random.rand(1)*(self.target_locations[self._location][3]-self.target_locations[self._location][2]) + self.target_locations[self._location][2]) # Random contr. on target x
            self._target_location[1] += np.float32(np.random.rand(1)*(self.target_locations[self._location][5]-self.target_locations[self._location][4]) + self.target_locations[self._location][4]) # Random contr. on target y

        # RANDOM LEVEL 6 - Path planning mode
        if (self._randomize_env_level == 6):
            # The location is already set
            self._target_location = np.array([self.waypoints_locations[self._which_waypoint][0], self.waypoints_locations[self._which_waypoint][1]], dtype=np.float32) # Base position
            self._target_location[0] += np.float32(np.random.rand(1)*(self.waypoints_locations[self._which_waypoint][3]-self.waypoints_locations[self._which_waypoint][2]) + self.waypoints_locations[self._which_waypoint][2]) # Random contr. on target x
            self._target_location[1] += np.float32(np.random.rand(1)*(self.waypoints_locations[self._which_waypoint][5]-self.waypoints_locations[self._which_waypoint][4]) + self.waypoints_locations[self._which_waypoint][4]) # Random contr. on target y
            
    def randomize_robot_location(self):
        ## This method randomizes robot's initial position based on self._randomize_env_level (0, 1, 2, 3, 4, 5)

        # RANDOM LEVEL 0 and 2 - Fixed position
        if (self._randomize_env_level == 0) or (self._randomize_env_level == 2):
            position_x = float(self._initial_agent_location[0])
            position_y = float(self._initial_agent_location[1])
            angle = float(self._initial_agent_location[2])
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        # RANDOM LEVEL 1, 3 and 6 - Single location randomization
        if (self._randomize_env_level == 1) or (self._randomize_env_level == 3) or (self._randomize_env_level == 6):
            # This method randomizes robot's initial position in a simple way
            position_x = float(1) + float(np.random.rand(1)*2-1) # Random contribution [-1,1]
            position_y = float(16) + float(np.random.rand(1) - 0.5) # Random contribution [-0.5,0.5]
            angle = float(math.radians(-90) + math.radians(np.random.rand(1)*60-30))
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        # RANDOM LEVEL 4 - Door test
        if (self._randomize_env_level == 4):
            # This resets the robot position in front of a door 6.7, 11.2
            position_x = float(6.7) + float(np.random.rand(1)*0.2 - 0.1) # Random contribution [-0.1,0.1]
            position_y = float(11.2) + float(np.random.rand(1)*0.3) # Random contribution [0,0.3]
            angle = float(math.radians(90) + math.radians(np.random.rand(1)*30-15)) # Random contribution [-15,15]
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        # RANDOM LEVEL 5 - Multiple location randomization
        if (self._randomize_env_level == 5):
            # Randomly decides which location to pick for spawning
            self._location = np.random.randint(0,len(self.robot_locations))
            # Here we set all the coordinates
            position_x = float(self.robot_locations[self._location][0]) + float(np.random.rand(1)*(self.robot_locations[self._location][4] - self.robot_locations[self._location][3]) + self.robot_locations[self._location][3])
            position_y = float(self.robot_locations[self._location][1]) + float(np.random.rand(1)*(self.robot_locations[self._location][6] - self.robot_locations[self._location][5]) + self.robot_locations[self._location][5])
            angle = float(math.radians(self.robot_locations[self._location][2]) + math.radians(np.random.rand(1)*(self.robot_locations[self._location][8] - self.robot_locations[self._location][7]) + self.robot_locations[self._location][7]))
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        return [position_x, position_y, orientation_z, orientation_w]
    
    def compute_rewards(self, info):
        # This method computes the reward of the step

        ## Simple reward
        if self._reward_method == 0:
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

        ## Heuristic reward
        elif self._reward_method == 1:
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
        elif self._reward_method == 2:
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
            
        return reward
    
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
        # Linear velocity can also be asymmetric
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
        self.destroy_client(self.client_state)
        self.destroy_publisher(self.action_pub)
        self.destroy_subscription(self.pose_sub)
        self.destroy_subscription(self.laser_sub)
        self.destroy_node()