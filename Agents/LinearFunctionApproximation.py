import random
import numpy as np
import copy
from Agents.generic_agents import Agent
from copy import deepcopy


class Linear(Agent):
    def __init__(self, env, num_episodes, gamma, epsilon, alpha):
        super(Linear, self).__init__(env, num_episodes, gamma)
        # initialize weight and bias
        self.w = np.zeros([10])
        self.bias = 1
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore action space using greedypolicy
        else:
            action = self.choose_best_action(state)  # Exploit learned values, take the best
        return action

    # update parameters
    def choose_best_action(self, state):
        if self.linear_calc(state, 0) > self.linear_calc(state, 1):
            return 0
        else:
            return 1

    def update(self, state, action, new_state, reward, done, current_episode, episode_length):
        prediction = self.linear_calc(state, action)  # actions given state s
        new_action = self.choose_best_action(new_state)
        target = self.linear_calc(new_state, new_action)
        delta = reward + self.gamma * target - prediction
        print(target, prediction)
        print(delta)
        state = np.concatenate((state, np.array([1])), axis=0)
        feature = self.get_feature_from_state_action(state, action)
        TD_error = self.alpha * delta
        self.w = self.w + TD_error * feature
        #   self.bias = self.bias + TD_error
        #                    self.w[state, action] = self.w[state, action] - TD_error * v
        #                self.bias = self.bias - TD_error
        #            TD_error = self.alpha * (lqf_s_a - (reward + self.gamma * lqf_s_a_next))
        #            if TD_error != 0.0:
        #                for state,v in states: #i = row, a = column, v = value of the state
        #                    self.w[state, action] = self.w[state, action] - TD_error * v
        #                self.bias = self.bias - TD_error
        return (False or done)

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


'''        def update_weights(self, weights, g):
            for i in range(len(weights)):
                for param in weights[i].keys():
                    weights[i][param] += self.step_size * g[i][param]
'''
