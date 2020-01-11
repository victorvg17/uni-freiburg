import sys
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd
from mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def tt(ndarray):
  return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)

def soft_update(target, source, tau):
  #Implement this
  assert len(target) == len(source), "unequal legnth for target and source vectors"
  # [v_target = v_target*(1 - tau) + v_source*tau for v_target, v_source in zip(target, source)]
  for v_target, v_source in zip(target, source):
    v_target = v_target*(1 - tau) + v_source*tau
  raise NotImplementedError("Implement a function to slowly update the parameters of target by the parameters of source with step size tau")

def hard_update(target, source):
  #Implement this
  assert len(target) == len(source), "unequal legnth for target and source vectors"
  [v_target = v_source for v_target, v_source in zip(target, source)]
  raise NotImplementedError("Implement a function to completely overwrite the parameters of target by the parameters of source")

class Q(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
    super(Q, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.fc3(x)

class ReplayBuffer:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self, max_size):
    self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
    self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
    self._size = 0
    self._max_size = max_size

  def add_transition(self, state, action, next_state, reward, done):
    # Implement this
    raise NotImplementedError("Implement the method that adds a transition to the replay buffer")


  def random_next_batch(self, batch_size):
    # Implement this
    raise NotImplementedError("Implement the method that draws a random minibatch from the replay buffer")

class DQN:
  def __init__(self, state_dim, action_dim, gamma):
    self._q = Q(state_dim, action_dim)
    self._q_target = Q(state_dim, action_dim)
    
    self._q.cuda()
    self._q_target.cuda()
    
    self._gamma = gamma
    self._loss_function = nn.MSELoss()
    self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
    self._action_dim = action_dim

    self._replay_buffer = ReplayBuffer(1e6)

  def get_action(self, x, epsilon):
    u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
    r = np.random.uniform()
    if r < epsilon:
      return np.random.randint(self._action_dim)
    return u

  def train(self, episodes, time_steps, epsilon):
    stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))    
    
    for e in range(episodes):
      print("%s/%s"%(e+1, episodes))
      s = env.reset()
      for t in range(time_steps):
        a = self.get_action(s, epsilon)
        ns, r, d, _ = env.step(a)

        stats.episode_rewards[e] += r
        stats.episode_lengths[e] = t

        # Implement this

        if d:
          break

        s = ns

    return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":
  env = MountainCarEnv() #gym.make("MountainCar-v0")
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  dqn = DQN(state_dim, action_dim, gamma=0.99)

  episodes = 1000
  time_steps = 200
  epsilon = 0.2
  
  stats = dqn.train(episodes, time_steps, epsilon)

  plot_episode_stats(stats)

  for _ in range(5):
    s = env.reset()
    for _ in range(200):
      env.render()
      a = dqn.get_action(s, epsilon)
      s, _, d, _ = env.step(a)
      if d:
        break