import random
from collections import deque
from copy import deepcopy
import gym
from Agents.generic_agents import Agent
import numpy as np


class LogisticPolicy:
    def __init__(self, theta, alpha, gamma):
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

    def logistic(self, y):
        # definition of logistic function
        return 1 / (1 + np.exp(-y))

    def probs(self, x):
        # returns probabilities of two actions
        y = x @ self.theta
        prob0 = self.logistic(y)

        return np.array([prob0, 1 - prob0])

    def act(self, x):
        # sample an action in proportion to probabilities
        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)
        return action, probs[action]

    def grad_log_p(self, x):
        # calculate grad-log-probs
        y = x @ self.theta
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = - x * self.logistic(y)
        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        # dot grads with future rewards for each action in episode
        return grad_log_p.T @ discounted_rewards

    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards
        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob, action in zip(obs, actions)])
        assert grad_log_p.shape == (len(obs), 4)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # gradients times rewards
        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)

        # gradient ascent on parameters
        self.theta += self.alpha * dot


class ReinforceAgent(Agent):
    def __init__(self, env, num_episodes, gamma, epsilon, alpha, max_buff_size=1200, batch_size=60):
        super(ReinforceAgent, self).__init__(env, num_episodes, gamma, epsilon)
        obs_size = env.observation_space.low.size
        self.env = env
        self.action_space_size = env.action_space.n
        self.epsilon = epsilon
        self.alpha = alpha
        self.total_rewards = np.zeros(self.num_episodes)
        self.policy = LogisticPolicy(np.random.rand(4), self.alpha, self.gamma)
        self.states = None
        self.actions = None
        self.rewards = None
        self.init_buffers()

    def init_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def choose_action(self, state):
        return self.policy.act(state)

    def update(self, state, action, new_state, reward, done):
        self.store_transitions(state, action, reward, done)

    def store_transitions(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update_after_ep(self):
        self.policy.update(self.states, self.rewards, self.actions)
        self.init_buffers()
