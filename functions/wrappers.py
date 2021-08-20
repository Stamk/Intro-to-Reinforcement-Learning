import math
import numpy as np
import gym


class StateDiscretize(gym.ObservationWrapper):
    def __init__(self, env, stepsizes=[],low=None, high=None):
        super(StateDiscretize, self).__init__(env)
        self.stepsizes = stepsizes

        if high is not None:
            self.env.observation_space.high = np.array(high)
        # TODO low

        self.bins = np.empty(len(self.stepsizes), dtype=object)
        for i in range(len(self.stepsizes)):
            self.bins[i] = np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i],self.stepsizes[i] - 1)
        # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.observation_space = gym.spaces.MultiDiscrete(self.stepsizes)
        print(self.bins)

    def observation(self, observation):
        for i in range(0, self.observation_space.shape[0]):
            observation[i] = np.digitize(observation[i], self.bins[i])  # return observation as index for Q-table
        return tuple(observation.astype(int))


class ActionDiscretize(gym.ActionWrapper):
    def __init__(self, env, stepsizes=[], low=None, high=None):
        super(ActionDiscretize, self).__init__(env)
        self.stepsizes = stepsizes

        if high is not None:
            self.env.observation_space.high = np.array(high)
        # TODO low

        self.bins = np.empty(len(self.stepsizes), dtype=object)
        for i in range(len(self.stepsizes)):
            # TODO fix
            self.bins[i] = np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i],
                                       self.stepsizes[i] - 1)
        # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.observation_space = gym.spaces.MultiDiscrete(self.stepsizes)
        print(self.bins)

    def action(self, action):
        for i in range(0, self.observation_space.shape[0]):
            observation[i] = np.digitize(observation[i], self.bins[i])  # return observation as index for Q-table
            # TODO fix
        return tuple(observation.astype(int))
