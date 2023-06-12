import gym
import numpy as np
import pandas as pd
import sys
import os
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# sys.path.append("../")
from environment.env import SumoEnvironment
import traci

from torch import nn as nn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
   
import streamlit as st
import matplotlib.pyplot as plt
import time

def run_method(method, user_input1, user_input2):
    env = SumoEnvironment(net_file=os.path.dirname(__file__)+ user_input1,
                            route_file=os.path.dirname(__file__)+ user_input2,
                            out_csv_name='output/waiting_times',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            max_depart_delay=0)

    if method == 'ppo':
        model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=128, n_epochs=20,
                    batch_size=256, clip_range=0.2, verbose=0)

        model.learn(total_timesteps=100000) 
        model.save("ppo_model.zip")
    
    elif method == 'a2c':
        model = A2C("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, n_steps=5, verbose=0)

        model.learn(total_timesteps=100000)
        model.save("a2c_model.zip")

    elif method == 'dqn':
        model = DQN('MlpPolicy', env, gamma=0.99, learning_rate=0.01, learning_starts=0, train_freq=1, target_update_interval=100, exploration_initial_eps=0.05,
                    exploration_final_eps=0.01, verbose=0)
        # Train the agent
        model.learn(total_timesteps=100000)
        model.save("dqn_model.zip")
    else:
        if method == "random":
            env.reset()
            for i in range(100000):
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                if done:
                    env.reset()
        elif method == "fixed":
            obs = env.reset()
            action = 0
            for i in range(50000):
                for j in range(2):
                    obs, reward, done, _ = env.step(action)
                    if done:
                        env.reset()
                action += 1
                if action > 3:
                    action = 0
            env.reset()

def main():
    user_input1 = str(st.text_input("Input the net file link"))
    st.write("You typed:", user_input1)
    user_input2 = str(st.text_input("Input the route file link"))
    st.write("You typed:", user_input2)

    st.sidebar.title("Method Selection")

    # Define the available methods as a list
    methods = ['ppo', 'a2c', 'dqn', 'fixed', 'random']

    # Create buttons for method selection
    method = st.sidebar.radio('You can select one of these methods:', methods)

    if st.sidebar.button('Run Method'):
            run_method(method, user_input1, user_input2)

if __name__ == '__main__':
    main()




    # def make_env(rank, user_input1, user_input2):
#     """
#     Utility function for multiprocessed env.
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = SumoEnvironment(net_file=os.path.dirname(__file__)+ user_input1,
#                             route_file=os.path.dirname(__file__)+ user_input2,
#                             out_csv_name='output/waiting_times{}'.format(rank),
#                             single_agent=True,
#                             use_gui=False,
#                             num_seconds=100000,
#                             max_depart_delay=0)
#         return env
#     return _init



####################### Create the vectorized environment #################################

# env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
# env = VecNormalize(env, norm_obs=True, norm_reward=True)

# model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=128, n_epochs=20,
#             batch_size=256, clip_range=0.2,verbose=0)

# model.learn(total_timesteps=800000) 
# model.save("ppo_model.zip")


# env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
# env = VecNormalize(env, norm_obs=True, norm_reward=True)
# model = A2C("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, n_steps=5, verbose=0)

# model.learn(total_timesteps=800000)
# model.save("a2c_model.zip")





######################### Saving and evaluating model ###################################

# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

# # Load the saved model
# model = PPO.load("ppo_model")

# # Evaluate the model's performance
# mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print("Mean reward:", mean_reward)

