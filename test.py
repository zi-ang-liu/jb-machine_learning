import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import torch
import gymnasium as gym
import numpy as np


class DQN(nn.Module):
    def __init__(self, env):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(env.observation_space.shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":

    # env
    env = gym.make("CliffWalking-v0")

    # Parameters
    n_buffer_capacity = 10000
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 0.001
    n_episodes = 1000
    epsilon = 0.1

    # Initialize empty replay memory
    memory = ReplayMemory(n_buffer_capacity)

    # Initialize DQN, DQN target and optimizer
    dqn = DQN(env).to(device)
    dqn_target = DQN(env).to(device)
    dqn_target.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

    for episode in range(n_episodes):

        # Reset environment
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False

        while not done:
            # Select action using epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = dqn(state).argmax().item()

            # Take action
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Save transition
            memory.push(state, action, next_state, reward)

            # Sample a batch of transitions
            Transitions = memory.sample(batch_size)
            batch = Transition(*zip(*Transitions))

            # Compute Q-values