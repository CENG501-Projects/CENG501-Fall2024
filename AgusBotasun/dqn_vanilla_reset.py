#!/usr/bin/env python3
"""
Vanilla DQN + Periodic Weight Reset
-----------------------------------
A single-agent DQN that periodically resets its Q-network parameters,
keeping the replay buffer intact.

 - This is the intermediate step: "DQN + Reset," to see if performance collapses.

Usage:
1) Train:
   python dqn_vanilla_reset.py
   (will write model after training)

2) Evaluate:
   python dqn_vanilla_reset.py --evaluate
   (loads model and runs e.g. 5 episodes, prints average reward)
"""

import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import argparse

import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Hyperparams
ENV_ID             = "MsPacmanNoFrameskip-v4"       #"Freeway-v4" "MsPacmanNoFrameskip-v4"
SEED               = 42

LR                 = 1e-4
GAMMA              = 0.99
BATCH_SIZE         = 32
REPLAY_BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE    = 10_000

TOTAL_TIMESTEPS    = 100_000

TRAIN_FREQUENCY    = 1        # steps between each training call
UPDATES_PER_STEP   = 1        # replay ratio
TARGET_UPDATE_FREQ = 1        # update target after each train step
MAX_GRAD_NORM      = 10

EPS_START          = 1.0
EPS_END            = 0.01
EPS_DECAY_STEPS    = 10_000    # epsilon decays over first 10k steps
LOG_INTERVAL       = 1_000

MODEL_PATH         = f"models/dqn_vanilla_reset_{ENV_ID}_seed{SEED}_rr{UPDATES_PER_STEP}.pth"
LOG_DIR            = f"runs/dqn_vanilla_reset_{ENV_ID}_seed{SEED}_rr{UPDATES_PER_STEP}"

# Vanilla Reset Parameters
RESET_FREQUENCY    = 40_000   # e.g., for Freeway and MsPacmanNoFrameskip from the table
RESET_DEPTH        = "full"  # "full" or "last2" or "last1". "full" is the default for vanilla reset !

# QNetworkAtari (with reset)
class QNetworkAtari(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Standard 3 conv layers -> 1 FC for Atari
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.reset_parameters("full")  # initial

    def forward(self, x):
        feats = self.features(x)
        feats = feats.reshape(feats.size(0), -1)  # use reshape to handle non-contiguous
        out   = self.fc(feats)
        return out

    def reset_parameters(self, reset_depth="full"):
        """
        Re-init some or all layers.
        'full'   => Re-init conv + fc
        'last2'  => Re-init last 2 layers in self.fc
        'last1'  => Re-init final linear in self.fc
        """
        def _init_module(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if reset_depth == "full":
            self.apply(_init_module)

        elif reset_depth == "last2":
            if len(self.fc) == 3:  # [Linear->ReLU->Linear]
                _init_module(self.fc[-1])  # final linear
                _init_module(self.fc[-3])  # linear before ReLU
            else:
                raise ValueError("Unexpected architecture for partial reset (last2).")

        elif reset_depth == "last1":
            if len(self.fc) == 3:
                _init_module(self.fc[-1])
            else:
                raise ValueError("Unexpected architecture for partial reset (last1).")

        else:
            raise ValueError("Unknown reset depth option.")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        obs       = np.stack(obs)
        acts      = np.array(acts, dtype=np.int64)
        rews      = np.array(rews, dtype=np.float32)
        next_obs  = np.stack(next_obs)
        dones     = np.array(dones, dtype=np.float32)
        return obs, acts, rews, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# DQN Agent + Vanilla Reset
class ResetDQNAgent:
    """
    DQN agent that:
      - Does a periodic "vanilla reset" of its Q-network
        while preserving the replay buffer
    """
    def __init__(self, n_actions, reset_depth=RESET_DEPTH, writer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Main Q-net + target
        self.q_network  = QNetworkAtari(n_actions).to(self.device)
        self.target_net = QNetworkAtari(n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_network.state_dict())

        self.optimizer  = optim.Adam(self.q_network.parameters(), lr=LR)
        self.n_actions  = n_actions

        self.global_step = 0
        self.train_steps = 0
        self.writer      = writer
        self.reset_depth = reset_depth

    def select_action(self, obs_np, epsilon=0.05):
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.n_actions-1)

        # channels-last => channels-first
        obs_ch_first = np.transpose(obs_np, (2,0,1))
        obs_t = torch.from_numpy(obs_ch_first).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            qvals = self.q_network(obs_t)
            action = qvals.argmax(dim=1).item()
        return action

    def train_on_batch(self, replay_buffer):
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            return

        obs, acts, rews, next_obs, dones = replay_buffer.sample(BATCH_SIZE)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        acts_t     = torch.LongTensor(acts).unsqueeze(1).to(self.device)
        rews_t     = torch.FloatTensor(rews).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q_next      = self.target_net(next_obs_t)
            max_q_next  = q_next.max(dim=1, keepdim=True)[0]
            target      = rews_t + GAMMA*(1 - dones_t)*max_q_next

        current_q = self.q_network(obs_t).gather(1, acts_t)
        loss      = nn.SmoothL1Loss()(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        self.train_steps += 1
        if self.writer:
            self.writer.add_scalar("Loss", loss.item(), self.train_steps)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_network.state_dict())
        if self.writer:
            self.writer.add_scalar("Info/TargetNetUpdates", 1, self.train_steps)

    def reset_weights(self):
        """
        Reset the Q-network parameters using self.reset_depth,
        then copy them to the target net as well.
        """
        print(f"** Resetting Q-network with reset_depth='{self.reset_depth}' **")
        self.q_network.reset_parameters(self.reset_depth)
        self.target_net.load_state_dict(self.q_network.state_dict())

    def save_model(self, path="q_network.pth"):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="q_network.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only= True)
        self.q_network.load_state_dict(ckpt)
        self.update_target_net()
        print(f"Model loaded from {path}")

# Make single VecEnv
def make_vec_env(env_id=ENV_ID, seed=42, clip_reward=True):
    env = make_atari_env(env_id, n_envs=1, seed=seed, wrapper_kwargs={'clip_reward': clip_reward})
    env = VecFrameStack(env, n_stack=4)
    env.current_obs = None
    return env

# Training loop with resets
def run_training():
    writer = SummaryWriter(log_dir=LOG_DIR)
    env = make_vec_env(ENV_ID, SEED)
    n_actions = env.action_space.n

    agent = ResetDQNAgent(n_actions, reset_depth=RESET_DEPTH, writer=writer)
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    obs = env.reset()  # [1,84,84,4]
    env.current_obs = obs

    episode_reward  = 0.0
    episode_count   = 0
    rewards_history = []

    eps_decay_steps = EPS_DECAY_STEPS

    print(f"Starting DQN + Weight Reset on {ENV_ID} for {TOTAL_TIMESTEPS} steps, reset_freq={RESET_FREQUENCY}...")

    while agent.global_step < TOTAL_TIMESTEPS:
        # compute epsilon
        fraction = min(1.0, agent.global_step / eps_decay_steps)
        epsilon  = EPS_START + fraction*(EPS_END - EPS_START)
        epsilon  = max(EPS_END, epsilon)

        obs_np = env.current_obs
        n_envs = obs_np.shape[0]

        actions = []
        for i in range(n_envs):
            a = agent.select_action(obs_np[i], epsilon)
            actions.append(a)

        next_obs, rewards, dones, infos = env.step(actions)
        agent.global_step += n_envs

        # Store transitions
        for i in range(n_envs):
            replay_buffer.add(
                np.transpose(obs_np[i], (2,0,1)),
                actions[i],
                rewards[i],
                np.transpose(next_obs[i], (2,0,1)),
                bool(dones[i])
            )

        # If done
        if any(dones):
            episode_count += 1
            ep_rew = float(np.mean(rewards))
            episode_reward += ep_rew
            rewards_history.append(episode_reward)
            writer.add_scalar("EpisodeReward", episode_reward, episode_count)

            obs = env.reset()
            env.current_obs = obs
            episode_reward = 0.0
        else:
            episode_reward += float(np.mean(rewards))
            env.current_obs = next_obs

        # Train
        if (agent.global_step % TRAIN_FREQUENCY)==0:
            for _ in range(UPDATES_PER_STEP):
                agent.train_on_batch(replay_buffer)
                if agent.train_steps>0 and (agent.train_steps % TARGET_UPDATE_FREQ)==0:
                    agent.update_target_net()

        # Periodic reset
        if (agent.global_step>0) and (agent.global_step % RESET_FREQUENCY)==0:
            writer.add_scalar("Info/ResetHappened", 1, episode_count)
            agent.reset_weights()

        # Log epsilon
        writer.add_scalar("Epsilon", epsilon, agent.global_step)

        # Logging
        if (agent.global_step % LOG_INTERVAL)==0 and agent.global_step>0:
            last_10 = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else np.mean(rewards_history) if len(rewards_history)>0 else 0
            print(f"Step={agent.global_step} | Episodes={episode_count} | "
                  f"AvgRew(last10)={last_10:.2f} | Eps={epsilon:.3f}")

    env.close()
    writer.close()

    # Save
    agent.save_model(MODEL_PATH)

    final_10 = np.mean(rewards_history[-10:]) if len(rewards_history)>=10 else 0
    print(f"Training done. Final 10-ep average reward: {final_10:.2f}")

# Evaluate loaded model
def evaluate_model(model_path=MODEL_PATH, episodes=10, epsilon=0.05, render=True, print_actions=False):
    """
    Load a saved model and run for 'episodes' episodes
    with a given 'epsilon' (like 0.05 or 0.0 for purely greedy).
    Print average reward.
    """
    # Use a different seed from training seed !
    # Do not clip rewards for evaluation !
    env = make_vec_env(ENV_ID, SEED + 100, False)
    n_actions = env.action_space.n

    agent = ResetDQNAgent(n_actions, reset_depth=RESET_DEPTH, writer=None)
    agent.load_model(model_path)

    episode_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        env.current_obs = obs
        done = False
        ep_rew = 0.0

        episode_actions = []
        while not done:
            obs_np = env.current_obs
            if render:
              env.render("human")
            action = agent.select_action(obs_np[0], epsilon=epsilon)
            episode_actions.append(action)
            obs_next, reward, done_bool, info = env.step([action])
            ep_rew += float(reward[0])
            env.current_obs = obs_next

            done = bool(done_bool[0])

        if print_actions:
          print(f"Episode Actions: {episode_actions}")
        episode_rewards.append(ep_rew)

    env.close()
    avg_rew = np.mean(episode_rewards)
    print(f"Evaluation over {episodes} episodes, epsilon={epsilon}: avg reward={avg_rew:.2f}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved model.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon used during evaluation.")
    parser.add_argument("--render", type=bool, default=True, help="Render evaluations on screen.")
    parser.add_argument("--print_actions", type=bool, default=False, help="Prints the action of the model during an episode.")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(model_path=MODEL_PATH, episodes=args.episodes, epsilon=args.epsilon, 
                       render=args.render, print_actions=args.print_actions)
    else:
        run_training()
