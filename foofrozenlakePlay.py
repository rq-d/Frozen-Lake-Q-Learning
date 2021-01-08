import gym
import numpy as np
import time
import pickle, os

EPISODES = 50

#env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.make('FrozenLake-v0')

with open("frozenLake_qTable.pkl", 'rb') as f:
	Q = pickle.load(f)

def choose_action(state):
	action = np.argmax(Q[state, :])
	return action
wins = 0
# start
for episode in range(EPISODES):

	state = env.reset()
	print("*** Episode: ", episode)
	t = 0
	while t < 20:
		env.render()

		action = choose_action(state)

		state2, reward, done, info = env.step(action)

		state = state2

		if done:
			if reward == 1:
				wins += 1
				print('\n*******WIN********\n')
			break

		#time.sleep(1)
		os.system('cls')
#print('\n\n', Q)
print('% Win :', wins/EPISODES*100, '%')
