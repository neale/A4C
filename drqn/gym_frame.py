import gym
import numpy as np

env = gym.make('MsPacman-v0')

env.reset()

for x in xrange(1000):
	env.step(x%9)

	#mode='human' for graphic output
	#Otherwise grab RGB data with mode='rgb_array'
	#actually returns bgr not rgb according to documentation
	frame = np.array(env.render(mode='rgb_array',close=False))
