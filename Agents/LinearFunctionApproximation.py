import random
import numpy as np
from Agents.generic_agents import Agent


class Linear(Agent):
    def __init__(self, envs ,name,type, num_episodes, gamma,eps=0.1, lr=0.1, anneal_lr_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(Linear, self).__init__(envs,name,type, num_episodes, gamma)
        # initialize weight and bias
        self.w = np.zeros(self.test_env.action_space.nvec[0]*self.test_env.observation_space.shape[0]+self.test_env.action_space.nvec[0])
        self.bias = 1
        self.eps=eps

    def choose_action(self, state,env):
        if random.uniform(0, 1) < self.eps:
            action = env.action_space.sample()  # Explore action space using greedypolicy
        else:
            action = self.choose_best_action(state,env)  # Exploit learned values, take the best
        return action

    # update parameters
    def choose_best_action(self, state,env):
        if self.linear_calc(state, 0) > self.linear_calc(state, 1):
            return 0
        else:
            return 1

    def update(self, state, action, new_state, reward, done):
        prediction = self.linear_calc(state, action)  # actions given state s
        new_action = self.choose_best_action(new_state,env=self.train_env)
        target = self.linear_calc(new_state, new_action)
        delta = reward + self.gamma * target - prediction
        state = np.concatenate((state, np.array([1])), axis=0)
        feature = self.get_feature_from_state_action(state, action)
        TD_error = self.lr * delta
        self.w = self.w + TD_error * feature


    def linear_calc(self, state, action):
        product = 0.0
        state = np.concatenate((state, np.array([1])), axis=0)
        feature = self.get_feature_from_state_action(state, action)
        product = np.dot(self.w, feature)
        return product

    def get_feature_from_state_action(self, state, action):
        # feature vector
        feature = np.zeros(len(self.w))
        offset = action * len(state)
        for i in range(len(state)):
            feature[i + offset] = state[i]
        return feature
