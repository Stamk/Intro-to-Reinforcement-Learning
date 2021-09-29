import math
import numpy as np
import gym

class StateDiscretize(gym.ObservationWrapper):
    def __init__(self, env, stepsizes=[],low=None, high=None):
        super(StateDiscretize, self).__init__(env)
        self.Statestepsizes = stepsizes

        if high is not None:
            self.env.observation_space.high = np.array(high)
        if low is not None:
            self.env.observation_space.low = np.array(-low)

        self.Statebins = np.empty(len(self.Statestepsizes), dtype=object)
        for i in range(len(self.Statestepsizes)):
            self.Statebins[i] = np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i],self.Statestepsizes[i])
        # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.observation_space = gym.spaces.MultiDiscrete(self.Statestepsizes)

    def observation(self, observation):
        for i in range(0, self.observation_space.shape[0]):
            observation[i] = np.digitize(observation[i], self.Statebins[i])-1  # return observation as index for table
        return tuple(observation.astype(int))


class ActionDiscretize(gym.ActionWrapper):
    def __init__(self, env, stepsizes=[], low=None, high=None):
        super(ActionDiscretize, self).__init__(env)
        self.Actionstepsizes = stepsizes

        if high is not None:
            self.env.action_space.high = np.array(high)
        if low is not None:
            self.env.action_space.low = np.array(low)

        self.Actionbins = np.empty(len(self.Actionstepsizes), dtype=object)
        #TODO fix Actionbins for more dimensions
        self.Actionbins = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0],self.Actionstepsizes[0])        # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(self.Actionstepsizes)

    def action(self, action):
        #TODO fix actions for more dimensions
        action = np.digitize(action, self.Actionbins)-1  # return action as index for table
        return action

    def step(self, action):
        return self.env.step((self.Actionbins[action]))

    def unwrappedaction(self,action):
        return self.Actionbins[action]