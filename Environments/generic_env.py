import math
import numpy as np
import gym


class Discretize(gym.ObservationWrapper):
    def __init__(self, env, stepsizes=[],start_date=" ",end_date=" ",Dt=" ",database=" "):
        super(Discretize, self).__init__(env)
        self.stepsizes = stepsizes
        self.start_date=start_date
        self.end_date=end_date
        self.Dt=Dt
        self.database=database
        if self.env.spec.id == 'CartPole-v0' or env.spec.id == 'CartPole-v1':
            self.env.observation_space.high[1] = 0.5
            self.env.observation_space.low[1] = -0.5
            self.env.observation_space.low[3] = - math.radians(50)
            self.env.observation_space.high[3] = math.radians(50)
        # self.stepsizes = tuple(self.stepsize for _ in range(self.env.observation_space.shape[0]))
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

    def NumOfStates(self):
        return np.sum(self.observation_space.nvec)
