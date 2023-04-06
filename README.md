# Hospitalbot path planning

> An infrastructure to train RL agents for a motion planning problem in an indoor environment (built with ROS2, Gazebo, OpenAI Gym, and Stable Baselines3).

## Description
This repository contains an application using ROS2 Humble, Gazebo, OpenAI Gym and Stable Baselines3 to train reinforcement learning agents which generate a feasible sequence of motion controls for a robot with a differential drive and a LIDAR to solve a path planning problem.

![hospitalbot-gif](.images/hospitalbot.gif)

The robot employed is a Pioneer 3AT with 4-wheel differential drive and a 180° laser for obstacle detection. The LIDAR collects 61 distance measurements that can range from 0.08 to 10 meters.

This repository includes the following elements:
* The 3D simulation environment of the hospital with the robot.
* A working Gym environment to train RL agents for the motion planning problem.
* Trained agents capable of reaching short and long distance targets inside the hospital environment while avoiding obstacles more than 80% of the time.

The application should be considered as a practical example for training reinforcement learning models using the open-source software previously mentioned. It is my first project with ROS2, the code is most likely not perfect, but it worked for my purpose.

## Current status
As this project was developed only for my Master's degree thesis, do not expect many updates over time. I published my work in the hope that I could help someone like me who is starting from scratch with ROS2 applications.

More information about the project can be found inside my thesis, which is available here (NOT AVAILABLE YET). I hope it helps :).

## Table of contents
- [Installation](#installation)
- [Getting started](#getting-started)
    - [Run a random agent](#run-a-random-agent)
    - [Run a trained agent](#run-a-trained-agent)
    - [Train a new agent](#train-a-new-agent)
    - [Hyperparameters tuning](#hyperparameters-tuning) (WORK IN PROGRESS)
    - [Re-train an existent agent](#re-train-an-existent-agent) (WORK IN PROGRESS)
- [References](#references)

## Installation
### Prerequisites
* ROS2 Humble or ROS2 Foxy (with Ubuntu 22.04 LTS or Ubuntu 20.04 LTS, respectively) - [install ROS2 Humble](https://docs.ros.org/en/humble/Installation.html);
* An already configured ROS2 workspace - [ROS2 workspace creation tutorial](https://www.youtube.com/watch?v=3GbrKQ7G2P0);
* Gazebo integration for ROS2 - [install gazebo_ros_pkgs](http://classic.gazebosim.org/tutorials?tut=ros2_installing&cat=connect_ros);
* Stable Baselines3 (includes also Gym) - [install Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html);
* Tensorboard - [install Tensorboard with pyp](https://pypi.org/project/tensorboard/);
* Optuna (for hyperparameters tuning) - [install Optuna](https://optuna.org/#installation);
### Step-by-step installation guide
First of all, clone this repository inside the src folder of your ROS2 workspace (replace `ros2_ws` with the name of your ROS2 workspace):
```
cd ~/ros2_ws/src
git clone https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning.git
```
You will need to copy all the files inside the `models` and `photos` folders inside the `~/.gazebo` repository. However, if you have never run Gazebo, the `~/.gazebo/models` and `~/.gazebo/photos` folders might not exist yet. To create them, simply launch the Gazebo empty world:
```
gazebo
```
Now, close the Gazebo window and the folders should have been created. Copy the files using these commands (replace `ros2_ws` with the name of your ROS2 workspace):
```
cd ~/ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner
cp -r models/. ~/.gazebo/models
cp -r photos/. ~/.gazebo/photos
```
At this point, build your ROS2 workspace to effectively install the package (replace `ros2_ws` with the name of your ROS2 workspace).
```
cd ~/ros2_ws
colcon build --packages-select hospital_robot_spawner
```
To check that everything is working, try to launch the hospital world.
```
ros2 launch hospital_robot_spawner gazebo_world.launch.py
```
The hospital world should be opened with the Gazebo application. Also the Pioneer 3AT should appear.

## Getting started
The application is quite complicated because it includes various modes. For this reason, before running anything, often the main scripts must be edited. I am planning on building a parameters file where all the specifics can be changed so that all the other files can remain untouched.

### Run a random agent
Running a random agent helps to understand the basic concepts of the Gym environment developed. Before running the commands, some files need to be edited.

* Open the "hospitalbot_env.py" file, search the `self._randomize_env_level` attribute of the **HospitalBotEnv** class and set one mode between the ones available (recommended: 0, 3, or 5). Also, make sure that the `self._visualize_target` attribute is set to True, otherwise the target will not be visualized. **Save the file at the end**.
* Edit the "start_training.py" file as follows. Search the `self._training_mode` attribute of the **TrainingNode** classe and assign the "random_agent" string to it. Below in the code, the number of episode to be simulated can be set (it should be 10 by default). **Save the file at the end**.
* Now, build your package again (replace `ros2_ws` with the name of your ROS2 workspace): `cd ~/ros2_ws;
colcon build --packages-select hospital_robot_spawner`.

After it is all set, launch the world file:
```
ros2 launch hospital_robot_spawner gazebo_world.launch.py
```
Finally, open another terminal and run the random agent script.
```
ros2 launch hospital_robot_spawner start_training.launch.py
```

### Run a trained agent
There are various trained agents inside the `rl_models` folder. To test one of them, some files need to be edited first.

* Edit the "trained_agent.py" script as follows. Find the `trained_model_path` variable and replace the last element of the `os.path.join` with the name of the desired agent (e.g., PPO_risk_seeker.zip). Save the file at the end.
* Edit the "hospitalbot_env.py" file to make sure that the correct mode is selected. Search the `self._randomize_env_level` attribute of the **HospitalBotEnv** class and pick one of the listed modalities (e.g., 6). Also, make sure that the `self._visualize_target` attribute is set to True, otherwise the target will not be visualized. Save the file at the end.
* Now, build your package again (replace `ros2_ws` with the name of your ROS2 workspace): `cd ~/ros2_ws;
colcon build --packages-select hospital_robot_spawner`.

After it is all set, launch the world file:
```
ros2 launch hospital_robot_spawner gazebo_world.launch.py
```
Finally, open another terminal and run the trained agent.
```
ros2 launch hospital_robot_spawner trained_agent.launch.py
```
### Train a new agent
* Firstly, edit the "hospitalbot_env.py" file. Set the `self._randomize_env_level` attribute to "5" as this mode implements the best setting to train an agent. Also, make sure that the `self._visualize_target` attribute is set to False, target visualization slows down the training significantly.
* Open the "start_training.py" file. Set the `self._training_mode` attribute to "training". Scroll down to the training section of the code inside the `elif node._training_mode == "training"` condition. Here, specify the number of total timesteps for which the agent is trained, and set the name of the agent file and the log file as you please.
* Now, build your package again (replace `ros2_ws` with the name of your ROS2 workspace): `cd ~/ros2_ws;
colcon build --packages-select hospital_robot_spawner`.

Once the changes are made, launch the training world. This launch file will not launch the graphical interface of Gazebo to make the training faster.
```
ros2 launch hospital_robot_spawner headless_world.launch.py
```
Finally, open another terminal and start the training using the command below.
```
ros2 launch hospital_robot_spawner start_training.launch.py
```
If you want to monitor the training process, open another terminal and type these commands (replace `ros2_ws` with the name of your ROS2 workspace).
```
cd ~/ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner
tensorboard --logdir=logs
```
At the end of the terminal there should be a link that you can open to visualize the tensorboard toolkit.

### Hyperparameters tuning
WORK IN PROGRESS

### Re-train an existent agent
WORK IN PROGRESS

## References
- **Pioneer 3AT model**: [link](https://github.com/dawonn/ros-pioneer3at)
- **Hospital world models**: [link](https://github.com/aws-robotics/aws-robomaker-hospital-world)
