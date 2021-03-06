import math
import numpy as np
import gym


class Discretize(gym.ObservationWrapper):
    def __init__(self, env, stepsizes=[]):
        super(Discretize, self).__init__(env)
        self.stepsizes = stepsizes
        if self.env.spec.id == 'CartPole-v0' or env.spec.id == 'CartPole-v1':
         self.env.observation_space.high[1] = 0.5
         self.env.observation_space.low[1] = -0.5
         self.env.observation_space.low[3] = - math.radians(50)
         self.env.observation_space.high[3] = math.radians(50)
      # self.stepsizes = tuple(self.stepsize for _ in range(self.env.observation_space.shape[0]))
        self.bins = np.empty(len(self.stepsizes), dtype=object)
        for i in range(len(self.stepsizes)):
            self.bins[i] = np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i], self.stepsizes[i] - 1)
      # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.observation_space = gym.spaces.MultiDiscrete(self.stepsizes)
        print(self.bins)

    def observation(self, observation):
        for i in range(0, self.observation_space.shape[0]):
          observation[i] = np.digitize(observation[i], self.bins[i])  #return observation as index for Q-table
        return tuple(observation.astype(int))

    def NumOfStates(self):
        return np.sum(self.observation_space.nvec)
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
