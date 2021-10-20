import math
import numpy as np
import gym

class PricesAddition(gym.ObservationWrapper):
    """
    It filters if observation space returns the extra states about Imbalance price and mean price
    """
    def __init__(self, env, prices_flag=True):
        super(PricesAddition, self).__init__(env)
        self.flag = prices_flag
        if not prices_flag: self.observation_space = gym.spaces.Box(env.observation_space.low[0:3],
                                                                    env.observation_space.high[0:3], dtype=np.float32)

    def observation(self, observation):
        if not self.flag: observation = observation[0:3]
        return observation

class StateDiscretize(gym.ObservationWrapper):
    """
    Discretization of a continuous observation space. It makes an observation space divided in the given step sizes.
    Note: length of the list with step sizes must be equal with the dimensions of observation space. Number of step sizes can be different for each dimension.
    For example, if stepsizes=[60,60,60,60] it returns 60 possible discrete observations for each of the four different observations.
    An empty list, returns observations unchanged.
    """
    def __init__(self, env, stepsizes=[], low=None, high=None):
        super(StateDiscretize, self).__init__(env)
        self.state_step_sizes = stepsizes*self.num_stack
        self.state_bins = np.empty(len(self.state_step_sizes), dtype=object)

        if high is not None:
            self.env.observation_space.high = np.array(high)
        if low is not None:
            self.env.observation_space.low = np.array(-low)
        if len(self.state_step_sizes) == 0:
            pass
        else:
            self.create_state_bins()
            self.observation_space = gym.spaces.MultiDiscrete(self.state_step_sizes)

    def create_state_bins(self):
        for i in range(len(self.state_step_sizes)):
            self.state_bins[i] = np.linspace(self.env.env.observation_space.low[0][i%2], self.env.env.observation_space.high[0][i%2],
                                             self.state_step_sizes[i%2])

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
    """
    Discretization of a continuous action space. It makes an action space divided in the given step sizes.
    Note: length of the list with step sizes must be equal with the dimensions of action space. Number of step sizes can be different for each dimension.
    For example, if stepsizes=[5,5,5] it returns 5 possible discrete actions for each of the three different actions.
    An empty list, returns actions unchanged.
    """
    def __init__(self, env, stepsizes=[], low=None, high=None):
        super(ActionDiscretize, self).__init__(env)
        self.action_step_sizes = stepsizes

        if high is not None:
            self.env.action_space.high = np.array(high)
        if low is not None:
            self.env.action_space.low = np.array(low)
        if len(self.action_step_sizes) == 0:
            pass
        else:
            self.create_action_bins()
            self.action_space = gym.spaces.MultiDiscrete(self.action_step_sizes)

    def create_action_bins(self):
        self.action_bins = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0],
                                       self.action_step_sizes[0])

    def action(self, action):
        if len(self.action_step_sizes) == 0:
            return action
        else:
            return self.discretize_action(action)

    def discretize_action(self, action):
        action = np.digitize(action, self.action_bins) - 1  # return action as index for table
        return action

    def step(self, action):
        if len(self.action_step_sizes) == 0:
            return self.env.step(action)
        else:
            return self.env.step((self.action_bins[action]))

    def unwrappedaction(self, action):
        if len(self.action_step_sizes) == 0:
            return action
        else:
            return self.action_bins[action]
