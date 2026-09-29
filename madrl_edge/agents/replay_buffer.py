"""
Shared replay buffer storing joint (all-agent) transitions, needed for the
centralized critic in CTDE.
"""
import random
import numpy as np
from collections import deque


class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, action_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_n, action_n, reward_n, next_obs_n, done_n):
        """
        obs_n, next_obs_n: list[num_agents] of obs_dim arrays
        action_n: list[num_agents] of action_dim arrays
        reward_n: list[num_agents] of floats
        done_n: list[num_agents] of bools
        """
        self.buffer.append((obs_n, action_n, reward_n, next_obs_n, done_n))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_n, action_n, reward_n, next_obs_n, done_n = zip(*batch)

        obs_n = np.array(obs_n, dtype=np.float32)             # (B, N, obs_dim)
        action_n = np.array(action_n, dtype=np.float32)       # (B, N, action_dim)
        reward_n = np.array(reward_n, dtype=np.float32)       # (B, N)
        next_obs_n = np.array(next_obs_n, dtype=np.float32)   # (B, N, obs_dim)
        done_n = np.array(done_n, dtype=np.float32)           # (B, N)

        return obs_n, action_n, reward_n, next_obs_n, done_n

    def __len__(self):
        return len(self.buffer)
