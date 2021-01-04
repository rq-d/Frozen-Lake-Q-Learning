# frozen-lake-ex1.py
import gym # loading the Gym library
import numpy as np
import time, pickle

env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 10
max_steps = 100

alpha = 0.81
gamma = 0.96 #discount factor

Q = np.zeros((env.observation_space.n, env.action_space.n))

def chooseAction(state):
    action = 0
    # make a random step most of the time
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    # choose the command with the highest probability
    else:
        action = np.argmax(Q[state,:])
    return action

def learn(state, state2, reward, action):
    Q[state, action] = Q[state,action] + alpha*( reward + gamma*np.max(Q[state2, :]) - Q[state, action])


for episode in range(total_episodes):
    state = env.reset()
    t = 0

    while t < max_steps:
        env.render()
        action = chooseAction(state)

        # move in the environment and return new state and reward
        state2, reward, done, info = env.step(action)

        # update Q function
        learn(state, state2, reward, action)

        state = state2

        t += 1

        if done:
            break

        #time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
