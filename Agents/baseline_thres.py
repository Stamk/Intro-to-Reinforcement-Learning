from Agents.generic_agents import Agent
import scipy.optimize
import numpy as np
from copy import deepcopy


class BaselineThresAgent(Agent):
    def __init__(self, envs, name,type,num_episodes, gamma, threshold=[-100, 100], lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=1):
        super(BaselineThresAgent, self).__init__(envs, name,type,num_episodes, gamma, lr, anneal_lr_param, threshold_lr_anneal,
                                             evaluate_every_n_episodes)
        self.threshold = threshold



    def choose_action(self, state,env):
        action = 0
        if env.unwrapped.state[-1] > self.threshold[1]:
            action = -1
        elif env.unwrapped.state[-1] < self.threshold[0]:
            action = 1
        return action

    def choose_best_action(self, state,env):
        return self.choose_action(state,env)