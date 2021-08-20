from copy import deepcopy
import numpy as np


class Agent:
    def __init__(self, env, num_episodes, gamma, lr=0.1, anneal_lr_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.anneal_lr_param = anneal_lr_param
        self.lr = lr
        self.threshold_lr_anneal = threshold_lr_anneal
        self.evaluate_every_n_episodes = evaluate_every_n_episodes
        self.results = None
        self.total_rewards = None

    def simulate(self, policy, train_flag=False):
        done = False
        cum_reward = 0.
        state = self.env.reset()
        while not done:
            action = policy(state)
            new_state, reward, done, info = self.env.step(action)
            if train_flag: self.update(state, action, new_state, reward, done)
            cum_reward += reward
            state = deepcopy(new_state)
        return cum_reward

    @staticmethod
    def linear_decay(val, param):
        val -= param
        return val

    @staticmethod
    def exp_decay(val, param):
        return val * param

    def anneal_lr(self, lr):
        return self.exp_decay(lr, self.anneal_lr_param)

    def train(self):
        self.total_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            episode_reward = self.simulate(policy=self.choose_action, train_flag=True)
            self.update_after_ep()
            self.total_rewards[i] = episode_reward
            if episode_reward > self.threshold_lr_anneal:
                self.lr = self.anneal_lr(self.lr)
            if i % self.evaluate_every_n_episodes == 0:
                print("episode", i)
                self.evaluate()

    def update_after_ep(self):
        pass

    def choose_action(self, state):
        return 1

    def choose_best_action(self, state):
        return 1

    def evaluate(self):
        episode_reward = self.simulate(policy=self.choose_best_action)
        print("Reward on evaluation %.2f" % episode_reward)

    def update(self, state, action, new_state, reward, done):
        pass

    def save(self,mean_of_episodes=50):
        mean_rewards = np.zeros(self.num_episodes)
        if self.total_rewards is not None:
            for i in range(self.num_episodes):
                mean_rewards[i] = np.mean(self.total_rewards[max(0, i - mean_of_episodes):(i + 1)])
        self.results = mean_rewards
