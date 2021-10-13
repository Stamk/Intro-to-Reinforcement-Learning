from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys

sys.path.append(path.abspath('../Documents/storage-balancing-env'))
from data_processing.database import load_db


class Agent:
    def __init__(self, envs, name, type, num_episodes, gamma, lr=0.1, anneal_lr_param=1., threshold_lr_anneal=100.,
                 evaluate_every_n_episodes=20):
        """
        :param env:
        :param num_episodes:
        :param gamma:
        :param lr:
        :param anneal_lr_param:
        :param threshold_lr_anneal:
        :param evaluate_every_n_episodes:
        """
        self.name = name
        self.type = type
        self.train_env, self.test_env = envs
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.anneal_lr_param = anneal_lr_param
        self.lr = lr
        self.threshold_lr_anneal = threshold_lr_anneal
        self.evaluate_every_n_episodes = evaluate_every_n_episodes
        self.episode_counter = 0
        self.results = None
        self.total_episodes_rewards = None
        self.total_train_rewards = []
        self.total_test_rewards = []
        self.train_states = []
        self.train_actions = []
        self.train_rewards = []
        self.test_states = []
        self.test_actions = []
        self.test_rewards = []

    def simulate(self, policy, env, train_flag=False, eval_flag=False):
        """
        :param policy:
        :param train_flag:
        :return:
        """
        done = False
        cum_reward = 0.
        state = env.reset()
        while not done:
            action = policy(state, env)
            new_state, reward, done, info = env.step(action)
            if not train_flag:
                self.store_transitions(state, action, reward, eval_flag=eval_flag)
            else:
                self.update(state, action, new_state, reward, done)
            cum_reward += reward
            state = deepcopy(new_state)
        return cum_reward

    def store_transitions(self, state, action, reward, eval_flag=False):
        """
        :param state:
        :param action:
        :param reward:
        :return:
        """
        if not eval_flag:
            self.train_states.append(self.train_env.unwrapped.state)
            self.train_actions.append(self.train_env.unwrappedaction(action))
            self.train_rewards.append(reward)
        else:
            self.test_states.append(self.test_env.unwrapped.state)
            self.test_actions.append(self.test_env.unwrappedaction(action))
            self.test_rewards.append(reward)

    def plot(self, exp_path):
        """
        :param exp_path:
        :return:
        """
        for param in ["train", "test"]:
            plt.figure()
            plt.title(self.name + param)
            states = getattr(self, param + "_states")
            ax1 = plt.subplot(311)
            ax1.set_title(param + " states")
            ax1.plot((np.reshape(states, (states.__len__(), states[0].shape[0])))[:, 0])
            ax2 = plt.subplot(312, sharex=ax1)
            ax2.set_title(param + "actions")
            actions = getattr(self, param + "_actions")
            ax2.plot(actions)
            ax3 = plt.subplot(313, sharex=ax1)
            ax3.set_title(param + " rewards")
            rewards = getattr(self, param + "_rewards")
            ax3.plot(rewards)
            env = getattr(self, param + "_env")
            plt.savefig('%s/%s agent of type %s on %s for %d episodes with learning rate %s and gamma %s for %s.png' % (
                exp_path, self.name, self.type, env.spec.id, self.num_episodes, self.lr, self.gamma, param))
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
        self.total_episodes_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            self.episode_counter = i
            episode_reward = self.simulate(policy=self.choose_action, train_flag=True, env=self.train_env)
            self.update_after_ep()
            self.total_episodes_rewards[i] = episode_reward
            if episode_reward > self.threshold_lr_anneal:
                self.lr = self.anneal_lr(self.lr)
            if i % self.evaluate_every_n_episodes == 0:
                print(self.name + " on episode ", i)
                self.evaluate()

    def update_after_ep(self):
        """
        :return:
        """
        pass

    def choose_action(self, state, env):
        """
        :param state:
        :return:
        """
        return 1

    def choose_best_action(self, state, env):
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
        train_episode_reward = self.simulate(policy=self.choose_best_action, env=self.train_env)
        self.total_train_rewards.append(train_episode_reward)
        print("Reward on train evaluation %.2f" % train_episode_reward)
        test_episode_reward = self.simulate(policy=self.choose_best_action, env=self.test_env, eval_flag=True)
        self.total_test_rewards.append(test_episode_reward)
        print("Reward on test evaluation %.2f" % test_episode_reward)

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

    def save_results(self, mean_of_episodes=50):
        """
        :param mean_of_episodes:
        :return:
        """
        mean_rewards = np.zeros(self.num_episodes)
        if self.total_episodes_rewards is not None:
            for i in range(self.num_episodes):
                mean_rewards[i] = np.mean(self.total_episodes_rewards[max(0, i - mean_of_episodes):(i + 1)])
        self.results = mean_rewards
