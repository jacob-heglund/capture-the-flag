# Author: Jacob Heglund
# This program features a deep Q-Learning Agent that is interfaced with the AI gym cartpole environment
# Heavily 'inspired by https://github.com/keon/deep-q-learning

#################################
# imports
# regular python stuff
import random
import gym
import numpy as np
from collections import deque

# torch neural net stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#################################
'''
class Agent():
    def __init__(self, sizeState, sizeAction):        
        self.sizeState = sizeState
        self.sizeAction = sizeAction
        self.memory = deque(maxlen=2000)
        
        # discount rate
        self.gamma = 0.95
        
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.model()

    def model(self):
        # Neural Net for Deep-Q learning Model
        #TODO: use the architecture from the double deep q-learning paper, use pytorch
        model = torch.nn.Sequential(
          #TODO: get the sizes for the convolutions
          torch.nn.Conv2d(),
          torch.nn.ReLU(),
          torch.nn.Conv2d(),
          torch.nn.Linear(H, D_out)
          ).to(device)

        
        model = Sequential()
        model.add(Dense(24, input_dim = self.sizeState, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.sizeAction, activation='linear'))
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
'''
EPISODES = 1000

env = gym.make('CartPole-v0')
sizeState = env.observation_space.shape[0]
sizeAction = env.action_space.n
for time in range(2000):
    env.render()

env.close()
'''
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    sizeState = env.observation_space.shape[0]
    sizeAction = env.action_space.n
    agent = Agent(sizeState, sizeAction)
    # agent.load("./save/cartpole-dqn.h5")
    done = False

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, sizeState])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, sizeState])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
'''