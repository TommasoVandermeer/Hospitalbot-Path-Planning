#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from hospital_robot_spawner.hospitalbot_env import HospitalBotEnv
from hospital_robot_spawner.hospitalbot_simplified_env import HospitalBotSimpleEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np

class TrainedAgent(Node):

    def __init__(self):
        super().__init__("trained_hospitalbot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

def main(args=None):
    rclpy.init()
    node = TrainedAgent()
    node.get_logger().info("Trained agent node has been created")

    # We get the dir where the models are saved
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
    trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'PPO_risk_seeker.zip')

    # Register the gym environment
    register(
        id="HospitalBotEnv-v0",
        entry_point="hospital_robot_spawner.hospitalbot_env:HospitalBotEnv",
        #entry_point="hospital_robot_spawner.hospitalbot_simplified_env:HospitalBotSimpleEnv",
        max_episode_steps=3000,
    )

    env = gym.make('HospitalBotEnv-v0')
    env = Monitor(env)

    check_env(env)

    episodes = 10

    # This is done to bypass the problem between using two different distros of ROS (humble and foxy)
    # They use different python versions, for this reason the action and observation space cannot be deserialized from the trained model
    # The solution is passing them as custom_objects, so that they won't be loaded from the model
    custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}

    # Here we load the rained model
    model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)

    # Evaluating the trained agent
    Mean_ep_rew, Num_steps = evaluate_policy(model, env=env, n_eval_episodes=100, return_episode_rewards=True, deterministic=True)

    # Print harvested data
    node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
    node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
    node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

    # Close env to print harvested info and destroy the hospitalbot node
    env.close()

    node.get_logger().info("The script is completed, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()