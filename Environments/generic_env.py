import math
import numpy as np
import gym


class Discretize(gym.ObservationWrapper):
    def __init__(self, env, stepsize=4, threshold=5):
        super(Discretize, self).__init__(env)
        self.stepsize = stepsize
        self.bins = np.linspace(self.env.observation_space.low, self.env.observation_space.high, self.stepsize + 1)
        obseravtions_shape = tuple(self.stepsize for _ in range(self.env.observation_space.shape[0]))
      # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.env.observation_space = gym.spaces.MultiDiscrete(obseravtions_shape)

    def observation(self, observation):
        for i in range(0, self.env.observation_space.shape[0]):
          observation[i] = np.digitize(observation[i], self.bins[:,i]) -1 #return observation as index for Q-table
        return tuple(observation.astype(int))

    def observation_nvec(self):
        return tuple(self.env.observation_space.nvec)

    def numOfStates(self):
        return np.sum(self.env.observation_space.nvec)
'''
    def observation(self, observation):
        for k in range(0, self.env.observation_space.shape[1]):
            for l in range(0, self.stepsize):
                if (observation[k] < self.env.observation_space[l + 1][k]) and (
                        observation[k] > self.env.observation_space[l][k]):
                    observation[k] = l
                    break
        return tuple(observation.astype(int))
'''

# class Scale(gym.ObservationWrapper):

#    def __init__(self, env):
#        super(Scale, self).__init__(env)

#    def observation(self, observation):
#        return observation - 1.


# class ReduceState(gym.ObservationWrapper):

#    def __init__(self, env):
#        super(ReduceState, self).__init__(env)

#    def observation(self, observation):
#       return observation[0]

"""
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = Discretize(env)
    done = False
    episode_reward = 0
    current_state = env.reset()
    counter = 0
    while not done:
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
    print("Terminal reward %f.4" % episode_reward)
"""
