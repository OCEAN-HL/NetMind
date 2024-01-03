from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*samples)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, batch_size, capacity):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.capacity = capacity
        self.replay_buffer = ReplayBuffer(capacity)
        self.model = DQN(state_dim, action_dim, hidden_dim)
        self.target_model = DQN(state_dim, action_dim, hidden_dim)
        self.update_target()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.steps = 0
        self.update_steps = 500

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state).argmax().item()
        return q_values

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)

    def learn(self):
        # Sample a batch from the replay buffer
        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)

        # Convert the batch to tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Compute Q values
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.model(next_state).max(1)[0]
            targets = reward + self.gamma * next_q_values

        # Update the Q values
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.steps += 1
        if self.steps % self.update_steps == 0:
            self.update_target()

    def train(
        self, env, observe, explore, train
    ):  # step 的输出有self.state.detach().numpy().astype(np.float32), reward, terminated, {}
        rewards = []
        state = env.reset()  # 第一步肯定是要reset的

        current_dir = os.path.abspath(os.path.dirname(__file__))

        while self.steps <= observe:
            process = "observe"
            action = torch.randint(0, 9, (1,)).item() # 在这里改，就可以实现不同的策略
            next_state, reward, terminated, _ = env.step(action)
            # print(f"next_state: {next_state}")
            if terminated == True:
                self.remember(state, action, reward, next_state)
                state = env.reset()  # 第一步肯定是要reset的
            else:
                state = next_state
                # print(f"state: {len(state)}")

            if self.steps % 1 == 0:
                status = "step {}/ process {}/ action {}/ reward {}/".format(self.steps, process, action, reward)
                print(status)
                f = open(current_dir + "/results/status_record", "a")
                # 'a' is attach in the end of file
                f.write(status + "\n")
                f.close()

            if self.steps % 1 == 0:
                rewards.append(reward)

            if self.steps % 1 == 0:
                data_reward = {"reward": rewards}
                data_reward = pd.DataFrame(data_reward)
                data_reward.to_csv(current_dir + "/results/reward")

            self.steps += 1
            # print("")
            # print("")
            # print("----------------------------------new------------------------------------")
            # print("----------------------------------step-----------------------------------")

        epsilon = self.epsilon
        while observe < self.steps <= (observe + explore + train):
            # epsilon -= (self.epsilon - 0.1) / explore
            epsilon -= (self.epsilon - 0.01) / explore
            process = None
            # Choose an action based on the exploration probability
            if np.random.rand() < epsilon:
                action = np.random.randint(self.action_dim)
                process = "explore"
            else:
                action = self.act(state)
                process = "train"

            # Take a step in the environment
            next_state, reward, terminated, _ = env.step(action)
            self.remember(state, action, reward, next_state)
            self.learn()
            if terminated == True:
                env.reset()
            else:
                state = next_state

            if self.steps % 1 == 0:
                status = "step {}/ process {}/ action {}/ reward {}/".format(self.steps, process, action, reward)
                print(status)
                f = open(current_dir + "/results/status_record", "a")
                # 'a' is attach in the end of file
                f.write(status + "\n")
                f.close()

            if self.steps % 1 == 0:
                rewards.append(reward)

            if self.steps % 1 == 0:
                data_reward = {"reward": rewards}
                data_reward = pd.DataFrame(data_reward)
                data_reward.to_csv(current_dir + "/results/reward")

            if self.steps % 1 == 0:
                episode_reward = {"reward": env.episode}
                episode_reward = pd.DataFrame(episode_reward)
                episode_reward.to_csv(current_dir + "/results/episode_reward")

            if self.steps % 1 == 0:
                episode_power_0 = {"power": env.power_recorder_0_all}
                episode_power_0 = pd.DataFrame(episode_power_0)
                episode_power_0.to_csv(current_dir + "/results/episode_power_0")
            
            if self.steps % 1 == 0:
                episode_power_1 = {"power": env.power_recorder_1_all}
                episode_power_1 = pd.DataFrame(episode_power_1)
                episode_power_1.to_csv(current_dir + "/results/episode_power_1")
            
            if self.steps % 1 == 0:
                episode_power_2 = {"power": env.power_recorder_2_all}
                episode_power_2 = pd.DataFrame(episode_power_2)
                episode_power_2.to_csv(current_dir + "/results/episode_power_2")
            
            if self.steps % 1 == 0:
                episode_power_3 = {"power": env.power_recorder_3_all}
                episode_power_3 = pd.DataFrame(episode_power_3)
                episode_power_3.to_csv(current_dir + "/results/episode_power_3")

            torch.save(self.model.state_dict(), current_dir + "/dqn_model.pth")
