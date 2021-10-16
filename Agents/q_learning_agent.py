import random
import gym
import numpy as np
from Agents.generic_agents import Agent


class QAgent(Agent):
    def __init__(self, envs, name, type, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=20):
        super(QAgent, self).__init__(envs, name, type, num_episodes, gamma, lr, anneal_lr_param,
                                     evaluate_every_n_episodes)
        assert isinstance(self.train_env.observation_space, gym.spaces.MultiDiscrete), "Train obs space not discretized"
        assert isinstance(self.test_env.observation_space, gym.spaces.MultiDiscrete), "Test obs space not discretized"
        assert isinstance(self.test_env.action_space, gym.spaces.MultiDiscrete), "Test action space not discretized"
        assert isinstance(self.train_env.action_space, gym.spaces.MultiDiscrete), "Train action space not discretized"
        # TODO all agents
        q_table_shape = np.concatenate((self.train_env.observation_space.nvec, self.train_env.action_space.nvec))
        self.q_table = np.zeros(q_table_shape)
        self.anneal_epsilon_param = anneal_epsilon_param
        self.eps = eps

    def choose_best_action(self, state, env):
        action = np.argmax(self.q_table[state])
        return action

    def choose_action(self, state, env):
        if random.uniform(0, 1) < self.eps:
            action = env.action_space.sample().item()  # Explore action space using greedypolicy
        else:
            action = self.choose_best_action(state, env)  # Exploit learned values, take the best
        return action

    def update(self, state, action, new_state, reward, done):
        new_q_value = np.max(self.q_table[new_state])
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + (self.gamma * new_q_value * (1 - done)))
        self.q_table[state][action] = updated_q_value

    def update_after_ep(self):
        self.eps = self.exp_decay(self.eps, self.anneal_epsilon_param)
        self.lr = self.exp_decay(self.lr, self.anneal_lr_param)
