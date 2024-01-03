import gym
import src.SFC.DRL.maze.maze
from src.SFC.DRL.NeuralNet import DQNAgent

env = gym.make("maze-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 128
# hidden_dim = 64 # training_5
lr = 0.0002 # training_0, training_1, training_2
# lr = 0.0004  # training_3, training_4, training_5, training_6
gamma = 0.99  # training_0, training_1, training_2, training_3, training_4, training_5
epsilon = 0.9  # training_0, training_1, training_2, training_3, training_4, training_5
# gamma = 0.98  # training_6
# epsilon = 0.8  # training_6
# batch_size = 30 # training_0, training_1, training_2
batch_size = 50  # training_3, training_4, training_5, training_6
buffer_capacity = 8000 # training_0, training_1, training_2, training_3
# buffer_capacity = 5000  # training_4, training_5, training_6
observe = 5000
explore = 4000  # training_0, training_1
# explore = 10000  # training_2
# explore = 8000  # training_3, training_4, training_5, training_6
train = 12000

# train
agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, batch_size, buffer_capacity)
agent.train(env, observe, explore, train)
