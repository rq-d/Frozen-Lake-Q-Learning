# frozen-lake-ex1.py
import gym # loading the Gym library
import numpy as np
import time, pickle

env = gym.make('FrozenLake-v0')
#env = gym.make('FrozenLake-v0', is_slippery=False)

epsilon = 0.7
total_episodes = 200
max_steps = 100

alpha = 0.05
gamma = 0.95 #discount factor

Q = np.zeros((env.observation_space.n, env.action_space.n))

def chooseAction(state):
    action = 0
    # make a random step most of the time for exploration
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    # choose the command with the highest probability
    else:
        action = np.argmax(Q[state,:])
    return action

def learn(state, state2, reward, action, done):
    if reward == 1:
        reward = 1
    elif done:
        reward = -1
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
        learn(state, state2, reward, action, done)

        state = state2

        t += 1

        if done:
            if reward == 1:
                print("\n\n ***************WINNNNN \n\n")
            break

        #time.sleep(0.1)

print('\n\nQ =', Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
