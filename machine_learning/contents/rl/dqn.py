import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class DQN(nn.Module):
    def __init__(self, dim_state, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(dim_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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

    Transition = namedtuple(
        "Transition", ("state", "action", "reward", "next_state", "done")
    )

    # env
    env = gym.make("CartPole-v1")

    # Parameters
    n_buffer_capacity = 10000
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 1e-4
    n_episodes = 1000
    epsilon = 0.1
    gamma = 0.99
    update_target = 10
    global_step = 0

    # writer
    writer = SummaryWriter()

    # Initialize empty replay memory
    memory = ReplayMemory(n_buffer_capacity)

    state, info = env.reset()
    dim_state = len(state) if isinstance(state, np.ndarray) else 1
    n_actions = env.action_space.n

    # Initialize DQN, DQN target and optimizer
    dqn = DQN(dim_state, n_actions).to(device)
    dqn_target = DQN(dim_state, n_actions).to(device)
    dqn_target.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

    for episode in range(n_episodes):

        # Reset environment
        state, info = env.reset()
        state = np.array(state)
        done = False
        episode_length = 0
        episode_reward = 0

        while not done:
            # Select action using epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn(torch.Tensor(state).to(device))
                    action = torch.argmax(q_values).item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Save transition
            memory.push(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Update global step
            global_step += 1

            # Update episode length and reward
            episode_length += 1
            episode_reward += reward

            # Update tensorboard
            if done:
                writer.add_scalar("reward", episode_reward, episode)
                writer.add_scalar("episode_length", episode_length, episode)

            # Sample a batch of transitions
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                # Convert lists of numpy arrays to single numpy arrays
                batch_state = np.array(batch.state)
                batch_next_state = np.array(batch.next_state)
                batch_action = np.array(batch.action)
                batch_reward = np.array(batch.reward)
                batch_done = np.array(batch.done)

                # Convert numpy arrays to tensors
                batch_state = torch.Tensor(batch_state).to(device)
                batch_action = torch.LongTensor(batch_action).to(device)
                batch_reward = torch.Tensor(batch_reward).to(device)
                batch_next_state = torch.Tensor(batch_next_state).to(device)
                batch_done = torch.Tensor(batch_done).to(device)

                # Compute Q-values
                q_values = (
                    dqn(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                )

                # Compute target Q-values
                with torch.no_grad():
                    target = batch_reward + gamma * torch.max(
                        dqn_target(batch_next_state), dim=1
                    ).values * (1 - batch_done)

                # Compute loss
                loss = F.mse_loss(q_values, target)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % update_target == 0:
            dqn_target.load_state_dict(dqn.state_dict())

    # Test the model
    n_run = 10

    for run in range(n_run):
        # test the model
        state, info = env.reset()
        state = np.array(state)
        done = False
        reward_record = []

        while not done:
            with torch.no_grad():
                q_values = dqn(torch.Tensor(state).to(device))
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_record.append(reward)
            done = terminated or truncated

            state = next_state

        print(f"Run {run}: {sum(reward_record)}")
