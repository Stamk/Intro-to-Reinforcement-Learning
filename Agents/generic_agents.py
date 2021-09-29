from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
sys.path.append(path.abspath('../Documents/storage-balancing-env'))
from data_processing.database import load_db


class Agent:
    def __init__(self, env,name,type, num_episodes, gamma, lr=0.1, anneal_lr_param=1.,threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        """
        :param env:
        :param num_episodes:
        :param gamma:
        :param lr:
        :param anneal_lr_param:
        :param threshold_lr_anneal:
        :param evaluate_every_n_episodes:
        """
        self.name=name
        self.type=type
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.anneal_lr_param = anneal_lr_param
        self.lr = lr
        self.threshold_lr_anneal = threshold_lr_anneal
        self.evaluate_every_n_episodes = evaluate_every_n_episodes
        self.episode_counter=0
        self.results = None
        self.total_rewards = None
        self.states=[]
        self.actions=[]
        self.rewards=[]

    def simulate(self, policy, train_flag=False):
        """
        :param policy:
        :param train_flag:
        :return:
        """
        done = False
        cum_reward = 0.
        state = self.env.reset()
        while not done:
            action = policy(state)
            new_state, reward, done, info = self.env.step(action)
            if not train_flag:
                self.store_transitions(state,action,reward)
            if train_flag: self.update(state, action, new_state, reward, done)
            cum_reward += reward
            state = deepcopy(new_state)
        return cum_reward

    def store_transitions(self,state,action,reward):
        """
        :param state:
        :param action:
        :param reward:
        :return:
        """
        self.states.append(self.env.unwrapped.state)
        self.actions.append(self.env.unwrappedaction(action))
        self.rewards.append(reward)

    def plot(self,exp_path):
        """
        :param exp_path:
        :return:
        """
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.set_title("States")
        ax1.plot((np.reshape(self.states,(self.states.__len__(),self.states[0].shape[0])))[:,0])
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title("Actions")
        ax2.plot(self.actions)
        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title("Rewards")
        ax3.plot(self.rewards)
        plt.savefig('%s/%s agent of type %s on %s for %d episodes with learning rate %s and gamma %s .png' % (exp_path, self.name,self.type, self.env.spec.id, self.num_episodes,self.lr,self.gamma))
        plt.show()


    @staticmethod
    def linear_decay(val, param):
        """
        :param val:
        :param param:
        :return:
        """
        val -= param
        return val

    @staticmethod
    def exp_decay(val, param):
        """
        :param val:
        :param param:
        :return:
        """
        return val * param

    def anneal_lr(self, lr):
        """
        :param lr:
        :return:
        """
        return self.exp_decay(lr, self.anneal_lr_param)

    def train(self):
        """
        :return:
        """
        self.total_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            self.episode_counter=i
            episode_reward = self.simulate(policy=self.choose_action, train_flag=True)
            self.update_after_ep()
            self.total_rewards[i] = episode_reward
            if episode_reward > self.threshold_lr_anneal:
                self.lr = self.anneal_lr(self.lr)
            if i % self.evaluate_every_n_episodes == 0:
                print("episode", i)
                self.evaluate()

    def update_after_ep(self):
        """
        :return:
        """
        pass

    def choose_action(self, state):
        """
        :param state:
        :return:
        """
        return 1

    def choose_best_action(self, state):
        """
        :param self:
        :param state:
        :return:
        """
        return 1

    def evaluate(self):
        """
        :param self:
        :return:
        """
        episode_reward = self.simulate(policy=self.choose_best_action)
        print("Reward on evaluation %.2f" % episode_reward)

    def update(self, state, action, new_state, reward, done):
        """
        :param state:
        :param action:
        :param new_state:
        :param reward:
        :param done:
        :return:
        """
        pass

    def save_results(self,mean_of_episodes=50):
        """
        :param mean_of_episodes:
        :return:
        """
        mean_rewards = np.zeros(self.num_episodes)
        if self.total_rewards is not None:
            for i in range(self.num_episodes):
                mean_rewards[i] = np.mean(self.total_rewards[max(0, i - mean_of_episodes):(i + 1)])
        self.results = mean_rewards
