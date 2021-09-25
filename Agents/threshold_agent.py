from Agents.generic_agents import Agent
import scipy.optimize
import numpy as np
from copy import deepcopy


class ThresholdAgent(Agent):
    def __init__(self, env, num_episodes, gamma, threshold=[50, 50], lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=20):
        super(ThresholdAgent, self).__init__(env, num_episodes, gamma, lr, anneal_lr_param, threshold_lr_anneal,
                                             evaluate_every_n_episodes)
        self.threshold = threshold
        self.mythreshold = threshold


    def choose_action(self, state):
        action = 0
        if self.env.unwrapped.state[-1] > self.threshold[1] and self.env.unwrapped.state[0] > \
                self.env.unwrapped.observation_space.low[0]:
            action = -1
        elif self.env.unwrapped.state[-1] < self.threshold[0] and self.env.unwrapped.state[0] < \
                self.env.unwrapped.observation_space.high[0]:
            action = 1
        return action

    def update_after_ep(self):
        result = self.nelder_mead_optimizer()
        self.threshold = deepcopy(result['x'])

    def update(self, state, action, new_state, reward, done):
        pass

    def train(self):
        """
        :return:
        """
        thres=self.threshold
        self.total_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            self.episode_counter=i
            episode_reward = self.simulate(policy=self.choose_action, train_flag=True)
            thres=deepcopy(self.update_after_my_ep(thres))
            print(thres)
            self.total_rewards[i] = episode_reward
            if episode_reward > self.threshold_lr_anneal:
                self.lr = self.anneal_lr(self.lr)
            if i % self.evaluate_every_n_episodes == 0:
                print("episode", i)
                self.evaluate()


    def objective(self, x):
        target=0
        eval_result=self.eval(x)
        result=target-eval_result
        return result

    def bfgs_optimizer(self):
        approximation = scipy.optimize.fmin_l_bfgs_b(lambda x: self.objective(x), 30, bounds=[(30, 100)], maxiter=20)
        return approximation

    def nelder_mead_optimizer(self,x):
        res = scipy.optimize.minimize(self.objective, x, method='Nelder-Mead', bounds=None)
        return res

    def newton_optimizer(self):
        range = np.arange(self.threshold[0], self.threshold[1])
        # root=scipy.optimize.newton(func=self.objective,x0=30 ,args=(range,))
        root = scipy.optimize.newton(lambda x: -self.mysimulate(x), x0=30)
        return root

    def choose_best_action(self, state):
        action = 0
        if self.env.unwrapped.state[-1] > self.threshold[1]:
            action = -1
        elif self.env.unwrapped.state[-1] < self.threshold[0]:
            action = 1
        return action


    def eval(self, threshold, train_flag=False):
        """
        :param policy:
        :param train_flag:
        :return:
        """
        mythreshold=threshold
        done = False
        cum_reward = 0.
        state = self.env.reset()
        while not done:
            action = 0
            if self.env.unwrapped.state[-1] > mythreshold[1]:
                action = -1
            elif self.env.unwrapped.state[-1] < mythreshold[0]:
                action = 1
            new_state, reward, done, info = self.env.step(action)
            if not train_flag:
                self.store_transitions(state,action,reward)
            if train_flag: self.update(state, action, new_state, reward, done)
            cum_reward += reward
            state = deepcopy(new_state)
        return cum_reward

    def update_after_my_ep(self,x):
        result = self.nelder_mead_optimizer(x)
        return result['x']