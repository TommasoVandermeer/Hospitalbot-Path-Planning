#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gym.envs.registration import register
from hospital_robot_spawner.hospitalbot_env import HospitalBotEnv
import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
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
    pkg_dir = '/home/tommaso/ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
    trained_model_path = os.path.join(pkg_dir, 'rl_models', 'PPO_norm_generalized_env.zip')

    # Register the gym environment
    register(
        id="HospitalBotEnv-v0",
        entry_point="hospital_robot_spawner.hospitalbot_env:HospitalBotEnv",
        #entry_point="hospital_robot_spawner.hospitalbot_simplified_env:HospitalBotSimpleEnv",
        max_episode_steps=300,
    )

    env = gym.make('HospitalBotEnv-v0')
    env = Monitor(env)


    episodes = 10

    # Here we load the rained model
    model = PPO.load(trained_model_path, env=env)

    # Evaluating the trained agent
    Mean_ep_rew, Num_steps = evaluate_policy(model, env=env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)

    node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
    node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
    node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

    """# Run the trained agent
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs) #, deterministic=True)
            obs, reward, done, info = env.step(action)
            node.get_logger().info("Observation: " + str(obs["agent"]))
            node.get_logger().info("Reward: " + str(reward))"""

    node.get_logger().info("The script is completed, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()