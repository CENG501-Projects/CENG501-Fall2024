#!/usr/bin/env python

FNAME = "atari_alien_rr4_plain_dqn_1"
import numpy as np

import torch
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor # required for minigrid
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed # may be required for seeded approaches
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# from gymnasium.wrappers import FrameStackObservation, ClipReward

# Enter parameters here
n_stack = 4 # run updates once every 4 frames (stack 4 environments for the model to train on)
eval_freq = 5000 # once every 5000 timesteps, evaluate the model
timesteps = 100000 # game timesteps
replay_ratio = 4 # run gradient calculations 4 times per step
reset_interval = 10000 # reset a part of the buffer at this timestep

# Our functions and methods

# Function to reset weights
def reset_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.reset_parameters()

# Custom callback to reset weights during training
class ResetWeightsCallback(BaseCallback):
    def __init__(self, reset_interval, verbose=0):
        super().__init__(verbose)
        self.reset_interval = reset_interval  # Number of steps between resets

    def _on_step(self) -> bool:
        # Reset weights every reset_interval steps
        if self.n_calls % self.reset_interval == 0: # n_calls inherited from BaseCallback
            # if self.verbose > 0:
            #     print(f"Resetting weights at step {self.n_calls}...")
            print(f"Policy weight reset at: {self.n_calls}")
            # Reset q_net and q_net_target
            self.model.policy.q_net.apply(reset_weights)
            self.model.policy.q_net_target.apply(reset_weights)
        return True

# Create gym environments
env = make_atari_env("AlienNoFrameskip-v4", n_envs=n_stack) #seed can be used here
env = VecFrameStack(env, n_stack= n_stack)
eval_env = make_atari_env("AlienNoFrameskip-v4", n_envs= n_stack) #seed can be used here, different than env's seed
eval_env = VecFrameStack(eval_env, n_stack= n_stack)

# External logger paths and policy setup
log_path = f"./logs/sb3_atari_dqn_1"
policy_kwargs = dict()

# Prepare callbacks
# callback frequencies are scaled to stack counts to match the given actual game timestep
eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path,
                             eval_freq=max(eval_freq // n_stack, 1), deterministic=True,
                             render=True)
# Create and attach the callback
reset_callback = ResetWeightsCallback(reset_interval=max(reset_interval // n_stack, 1), verbose=1)

callback_list = CallbackList([eval_callback, reset_callback])

# Model declaration
model = DQN(
    policy= "CnnPolicy", 
    env= env, 
    verbose= 1, 
    buffer_size= timesteps,
    learning_starts= 2000,
    tau= 0.005,
    train_freq= (1, "step"),
    gradient_steps= replay_ratio,
    target_update_interval= 1,
    policy_kwargs= policy_kwargs,
    tensorboard_log="./dqn_atari_logs",
    )
# TODO: need reset, reset frequency and reset depth
model.learn(
    total_timesteps=timesteps,
    callback=callback_list
    )

# Close environments and save model when done training
env.close()
eval_env.close()
model.save(f"./models/{FNAME}") 

# Optionally import the model if you already have trained one
# model = DQN.load(f"./models/{FNAME}", env= env) # use this cell if you already have a trained model

# Run the evaluation
eval_eps = 10
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_eps)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(eval_eps):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

print(f"mean_reward: {mean_reward}, std_reward:{std_reward}")
# Close the evaluation environment
vec_env.close()