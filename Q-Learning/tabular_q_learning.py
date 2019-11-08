'''
Filename: c:\dev\research\sandbox\Q-Learning\tabular_q_learning.py
Path: c:\dev\research\sandbox\Q-Learning
Created Date: Wednesday, October 3rd 2018, 9:38:01 pm
Author: Jacob Heglund

Copyright (c) 2018 Jacob Heglund
'''

import gym
gym.logger.set_level(40)
import subprocess
import numpy as np
import time

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

learningRate = 0.8
gamma = 0.95
numEpisodes = 6000

# a list of total rewards each episode
rList = []

for episode in range(numEpisodes):
    # render the environment every so often
    display = False
    if (episode != 0) and (episode % 5999 == 0):
        display = True

    state = env.reset()
    rewardTotal = 0
    # done tells when a particular episode ends (i.e. when your agent reaches the goal or dies)
    done = False
    j = 0

    # specify the number of actions to take before ending the episode
    while j < 99:
        j += 1
        # choose a greedy action picking from the Q-table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1.0/(episode+1.0)))

        # get new state and resard from env
        stateNext , reward, done, prob = env.step(action)
        if display:
            subprocess.call(["printf", "'\033c'"])
            print(episode)
            env.render()
            

            time.sleep(.1)
        # update Q-Table with results 
        Q[state, action] = Q[state, action] + learningRate * (reward+ gamma*np.max(Q[stateNext, :]) - Q[state, action])
        rewardTotal += reward
        state = stateNext

        if done == True:
            break
        
    rList.append(rewardTotal)

print("Score over time: " +  str(sum(rList)/numEpisodes))
print("Final Q-Table Values")
print(Q)

