#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gym.envs.registration import register
from hospital_robot_spawner.hospitalbot_env import HospitalBotEnv
from hospital_robot_spawner.hospitalbot_simplified_env import HospitalBotSimpleEnv
import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import optuna
from stable_baselines3.common.evaluation import evaluate_policy

class TrainingNode(Node):

    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Defines which action the script will perform "random_agent", "training", "retraining" or "hyperparam_tuning"
        self._training_mode = "retraining"

        # Get training parameters from Yaml file
        #self.test = super().get_parameter('test').value
        #self.get_logger().info("Test parameter: " + str(self.test))

def main(args=None):

    # Initialize the training node to get the desired parameters
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Create the dir where the trained RL models will be saved
    pkg_dir = '/home/tommaso/ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
    trained_models_dir = os.path.join(pkg_dir, 'rl_models')
    log_dir = os.path.join(pkg_dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # First we register the gym environment created in hospitalbot_env module
    register(
        id="HospitalBotEnv-v0",
        entry_point="hospital_robot_spawner.hospitalbot_env:HospitalBotEnv",
        #entry_point="hospital_robot_spawner.hospitalbot_simplified_env:HospitalBotSimpleEnv",
        max_episode_steps=300,
    )

    node.get_logger().info("The environment has been registered")

    #env = NormalizeReward(gym.make('HospitalBotEnv-v0'))
    env = gym.make('HospitalBotEnv-v0')

    # Sample Observation and Action space for Debugging
    #node.get_logger().info("Observ sample: " + str(env.observation_space.sample()))
    #node.get_logger().info("Action sample: " + str(env.action_space.sample()))

    # Here we check if the custom gym environment is fine
    check_env(env)
    node.get_logger().info("Environment check finished")

    # Now we create two callbacks which will be executed during training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=25000, best_model_save_path=trained_models_dir, n_eval_episodes=15)
    
    if node._training_mode == "random_agent":
        # N° Episodes
        episodes = 10
        ## Execute a random agent
        node.get_logger().info("Starting the RANDOM AGENT now")
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
                node.get_logger().info("Agent state: [" + str(info["distance"]) + ", " + str(info["angle"]) + "]")
                node.get_logger().info("Reward at step " + ": " + str(reward))
    
    elif node._training_mode == "training":
        ## Train the model
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=2279, gamma=0.9880614935504514, gae_lambda=0.9435887928788405, ent_coef=0.00009689939917928778, vf_coef=0.6330533453055319, learning_rate=0.00011770118633714448, clip_range=0.1482)
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=6400, gamma=0.9880614935504514, gae_lambda=0.9435887928788405, ent_coef=0.00009689939917928778, vf_coef=0.6330533453055319, learning_rate=0.00003770118633714448, clip_range=0.1482)
        # Execute training
        try:
            model.learn(total_timesteps=int(3000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="PPO_with_obstacles")
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/PPO_with_obstacles")
        # Save the trained model
        model.save(f"{trained_models_dir}/PPO_with_obstacles")
    
    elif node._training_mode == "retraining":
        ## Re-train an existent model
        node.get_logger().info("Retraining an existent model")
        # Path in which we find the model
        trained_model_path = os.path.join(pkg_dir, 'rl_models', 'PPO_trial_9.zip')
        # Here we load the rained model
        #custom_obs = {'learning_rate': 0.000003, 'ent_coef': 0.01}
        model = PPO.load(trained_model_path, env=env) #, custom_objects=custom_obs)
        # Execute training
        try:
            model.learn(total_timesteps=int(3000000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="PPO_trial_9")
        except KeyboardInterrupt:
            # If you notice that the training is sufficiently well interrupt to save
            model.save(f"{trained_models_dir}/PPO_trial_9_retrained")
        # Save the trained model
        model.save(f"{trained_models_dir}/PPO_trial_9_retrained")

    elif node._training_mode == "hyperparam_tuning":
        # Delete previously created environment
        env.close()
        del env
        # Hyperparameter tuning using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_agent, n_trials=10, n_jobs=1)
        # Print best params
        node.get_logger().info("Best Hyperparameters: " + str(study.best_params))

    # Shutting down the node
    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

def optimize_ppo(trial):
    ## This method defines the range of hyperparams to search fo the best tuning
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1), # Default: 0.5
    }

def optimize_ppo_refinement(trial):
    ## This method defines a smaller range of hyperparams to search fo the best tuning
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 14336), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.96, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 9e-4), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.15, 0.37), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.94, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.00001), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0.55, 0.65), # Default: 0.5
    }

def optimize_agent(trial):
    ## This method is used to optimize the hyperparams for our problem
    try:
        # Create environment
        env_opt = gym.make('HospitalBotEnv-v0')
        # Setup dirs
        PKG_DIR = '/home/tommaso/ros2_ws/src/Hospitalbot-Path-Planning/hospital_robot_spawner'
        LOG_DIR = os.path.join(PKG_DIR, 'logs')
        SAVE_PATH = os.path.join(PKG_DIR, 'tuning', 'trial_{}'.format(trial.number))
        # Setup the parameters
        #model_params = optimize_ppo(trial)
        model_params = optimize_ppo_refinement(trial)
        # Setup the model
        model = PPO("MultiInputPolicy", env_opt, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=150000)
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        # Close env and delete
        env_opt.close()
        del env_opt

        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -10000

if __name__ == "__main__":
    main()