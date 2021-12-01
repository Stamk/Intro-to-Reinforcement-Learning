from Agents.generic_agents import Agent
import scipy.optimize
import numpy as np
from copy import deepcopy


class ThresholdAgent(Agent):
    def __init__(self, envs, name,type,num_episodes, gamma, threshold=[0, 30], lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=20, method='Nelder-Mead'):
        super(ThresholdAgent, self).__init__(envs, name,type,num_episodes, gamma, lr, anneal_lr_param, threshold_lr_anneal,
                                             evaluate_every_n_episodes)
        self.threshold = threshold
        self.method = method

    def train(self):
        self.threshold = self.optimizer(self.threshold)
        self.total_rewards = np.zeros(self.num_episodes)
        self.evaluate()

    def optimizer(self, x):
        res = scipy.optimize.minimize(self.eval, x, method=self.method,options={'disp': True}, tol=0.0000001)
        print(res.x)
        return res.x

    def choose_best_action(self, state,env):
        action = 0
        if env.unwrapped.state[-1] > self.threshold[1]:
            action = -1
        elif env.unwrapped.state[-1] < self.threshold[0]:
            action = 1
        return action

    def eval(self, threshold):
        """
        :param policy:
        :param train_flag:
        :return:
        """
        # threshold[0] = threshold[0].clip(max=threshold[1])
        done = False
        cum_reward = 0.
        state = self.train_env.reset()
        while not done:
            action = 0
            if self.train_env.unwrapped.state[-1] > threshold[1]:
                action = -1
            elif self.train_env.unwrapped.state[-1] < threshold[0]:
                action = 1
            new_state, reward, done, info = self.train_env.step(action)
            cum_reward += reward
            state = deepcopy(new_state)
        return -cum_reward
