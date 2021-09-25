import random
import numpy as np
import copy
from Agents.generic_agents import Agent
from copy import deepcopy


class Linear(Agent):
    def __init__(self, env, num_episodes, gamma,eps=0.1, lr=0.1, anneal_lr_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(Linear, self).__init__(env, num_episodes, gamma)
        # initialize weight and bias
        self.w = np.zeros([10])
        self.bias = 1
        self.eps=eps

    def choose_action(self, state):
        if random.uniform(0, 1) < self.eps:
            action = self.env.action_space.sample()  # Explore action space using greedypolicy
        else:
            action = self.choose_best_action(state)  # Exploit learned values, take the best
        return action

    # update parameters
    def choose_best_action(self, state):
        if self.linear_calc(state, -1) > self.linear_calc(state, 1):
            return -1
        else:
            return 1

    def update(self, state, action, new_state, reward, done):
        prediction = self.linear_calc(state, action)  # actions given state s
        new_action = self.choose_best_action(new_state)
        target = self.linear_calc(new_state, new_action)
        delta = reward + self.gamma * target - prediction
        state = np.concatenate((state, np.array([1])), axis=0)
        feature = self.get_feature_from_state_action(state, action)
        TD_error = self.lr * delta
        self.w = self.w + TD_error * feature

    # return q(s, a; w) for a given state s and action a
    def linear_calc(self, state, action):
        product = 0.0
        state = np.concatenate((state, np.array([1])), axis=0)
        feature = self.get_feature_from_state_action(state, action)
        #            for state,v in states:
        #                product += v * self.w[state, action]
        product = np.dot(self.w, feature)
        return product

    def get_feature_from_state_action(self, state, action):
        # feature vector
        feature = np.zeros(len(self.w))
        offset = action * len(state)
        for i in range(len(state)):
            feature[i + offset] = state[i]
        # feature[-1]=self.bias
        return feature
