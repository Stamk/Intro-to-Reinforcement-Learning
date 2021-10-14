import math
import numpy as np
import gym


class PricesAddition(gym.ObservationWrapper):
    def __init__(self, env, prices_flag=True):
        super(PricesAddition, self).__init__(env)
        self.flag = prices_flag
        if not prices_flag: self.observation_space = gym.spaces.Box(env.observation_space.low[0:3],env.observation_space.high[0:3],dtype=np.float32)

    def observation(self, observation):
        if not self.flag: observation = observation[0:3]
        return observation


class StateDiscretize(gym.ObservationWrapper):
    def __init__(self, env, stepsizes=[], low=None, high=None):
        super(StateDiscretize, self).__init__(env)
        self.state_step_sizes = stepsizes
        self.state_bins = np.empty(len(self.state_step_sizes), dtype=object)

        if high is not None:
            self.env.observation_space.high = np.array(high)
        if low is not None:
            self.env.observation_space.low = np.array(-low)
        if len(self.state_step_sizes) == 0:
            pass
        else:
            self.create_statebins()
            self.observation_space = gym.spaces.MultiDiscrete(self.state_step_sizes)

    def create_statebins(self):
        for i in range(len(self.state_step_sizes)):
            self.state_bins[i] = np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i],
                                             self.state_step_sizes[i])

    def observation(self, observation):
        if len(self.state_step_sizes) == 0:
                return observation
        else:
                return self.discretize_observation(observation)

    def discretize_observation(self, observation):
        for i in range(0, self.observation_space.shape[0]):
            observation[i] = np.digitize(observation[i],
                                         self.state_bins[i]) - 1  # return observation as index for table
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
        # TODO fix Actionbins for more dimensions
        self.Actionbins = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], self.Actionstepsizes[
            0])  # self.env.observation_space = gym.spaces.Box(self.env.observation_space.low, self.env.observation_space.high,shape=(self.env.observation_space.shape[0], stepsize), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(self.Actionstepsizes)

    def action(self, action):
        # TODO fix actions for more dimensions
        action = np.digitize(action, self.Actionbins) - 1  # return action as index for table
        return action

    def step(self, action):
        return self.env.step((self.Actionbins[action]))

    def unwrappedaction(self, action):
        return self.Actionbins[action]
