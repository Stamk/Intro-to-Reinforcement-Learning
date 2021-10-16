from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, envs, name, type, num_episodes, gamma, lr=0.1, anneal_lr_param=1., threshold_lr_anneal=100.,
                 evaluate_every_n_episodes=20):
        """
        A generic agent which all agents can inherit with some basic common attributes and parameters for all agents
        :param envs: current working environment
        :param name: given name of current agent
        :param type: type of algorithm it uses
        :param num_episodes: number of episodes
        :param gamma: discount factor gamma
        :param lr: learning rate, alpha
        :param anneal_lr_param: parameter for learning rate decrease
        :param threshold_lr_anneal: over the value for the cumulative reward to start the learning rate decrease
        :param evaluate_every_n_episodes: number of episodes to make an evaluation
        """
        self.name = name
        self.type = type
        self.train_env, self.test_env = deepcopy(envs)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.anneal_lr_param = anneal_lr_param
        self.lr = lr
        self.threshold_lr_anneal = threshold_lr_anneal
        self.evaluate_every_n_episodes = evaluate_every_n_episodes
        self.episode_counter = 0
        self.total_episodes_rewards = None
        self.total_train_rewards = []
        self.total_test_rewards = []
        self.train_states = []
        self.train_actions = []
        self.train_rewards = []
        self.test_states = []
        self.test_actions = []
        self.test_rewards = []

    def train(self):
        """
        Train the agent for all the episodes by calling simulate function with respective arguments
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

    def simulate(self, policy, env, train_flag=False, eval_flag=False):
        """
        Makes a simulation of an episode for an environment and under a policy given

        Args:
            :param env: current environment to simulate
            :param eval_flag (bool) : set True for test environment evaluation otherwise train environment evaluation
            :param policy (function) : function that defines how the agent takes any actions
            :param train_flag (bool) : set True if simulate is on train mode

        Returns:
            :return: the cumulative reward of the episode
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

    def evaluate(self):
        """
        Evaluates agents on train and test environments by calling simulate
        """
        train_episode_reward = self.simulate(policy=self.choose_best_action, env=self.train_env)
        self.total_train_rewards.append(train_episode_reward)
        print("Reward on train evaluation %.2f" % train_episode_reward)
        test_episode_reward = self.simulate(policy=self.choose_best_action, env=self.test_env, eval_flag=True)
        self.total_test_rewards.append(test_episode_reward)
        print("Reward on test evaluation %.2f" % test_episode_reward)


    def store_transitions(self, state, action, reward, eval_flag=False):
        """
        Stores traces of all states, rewards and actions for current env and agent
        Args:
            :param state (tuple) : current state
            :param action (int) : current action taken
            :param reward (int) : reward of action in given state
            :param eval_flag (bool) :  set True to store in test environment otherwise in train environment
        """
        if not eval_flag:
            self.train_states.append(self.train_env.unwrapped.state)
            self.train_actions.append(self.train_env.unwrappedaction(action))
            self.train_rewards.append(reward)
        else:
            self.test_states.append(self.test_env.unwrapped.state)
            self.test_actions.append(self.test_env.unwrappedaction(action))
            self.test_rewards.append(reward)

    @staticmethod
    def linear_decay(val, param):
        """
        Computes the difference of a value by a parameter given
        Args:
            :param val: value of minuend
            :param param: value of subtrahend
        Return:
            :return: the subtraction of value by parameter
        """
        val -= param
        return val

    @staticmethod
    def exp_decay(val, param):
        """
        It decreases exponential a value given
        Args:
            :param val: the value to be decreased
            :param param: exponential decay parameter
        Return:
            :return: the new exponential decreased value of the value given
        """
        return val * param

    def anneal_lr(self, lr):
        """
        Args:
            :param lr: agent's current learning rate
        Return:
            :return: the new decreased value of learning rate
        """
        return self.exp_decay(lr, self.anneal_lr_param)

    def update_after_ep(self):
        """
        Makes all the required updates the agent needs after an episode terminates
        """
        pass

    def choose_action(self, state, env):
        """
        Takes an action according to policy for the current state of the environment given
        Args:
            :param state (tuple) : agent's current state
            :param env: agent's current env

        Returns:
            :return: action according to agent's policy
        """
        return 1

    def choose_best_action(self, state, env):
        """
        Takes the best possible action according for the current state of the environment given
        Args:
            :param state (tuple) : agent's current state
            :param env : agent's current env

        Returns:
            :return: the best action according to agent's policy
        """
        return 1

    def update(self, state, action, new_state, reward, done):
        """
        Makes all the required updates after the episode is finished

        Args:
            :param state (tuple) : agent's current state
            :param action (int) : an action provided by the agent
            :param new_state (tuple) :  agent's new state after taking the action
            :param reward (int) : amount of reward returned after action
            :param done (bool) : whether the episode has ended

        """
        pass
