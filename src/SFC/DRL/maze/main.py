import gym
import src.SFC.DRL.maze.maze
from src.SFC.DRL.NeuralNet import DQNAgent

env = gym.make("maze-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 128
lr = 0.0002 
gamma = 0.99  
epsilon = 0.9  
batch_size = 50  
buffer_capacity = 8000 
observe = 5000
explore = 4000  
train = 12000

# train
agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, batch_size, buffer_capacity)
agent.train(env, observe, explore, train)
