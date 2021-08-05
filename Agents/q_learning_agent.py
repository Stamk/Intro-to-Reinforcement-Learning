import random
import numpy as np
from Agents.generic_agents import Agent


class QAgent(Agent):

    def __init__(self, env, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(QAgent, self).__init__(env, num_episodes, gamma, lr, eps, anneal_lr_param,
                                     anneal_epsilon_param,
                                     threshold_lr_anneal, evaluate_every_n_episodes)
        self.q_table = np.zeros(np.concatenate((self.env.observation_space.nvec, [self.env.action_space.n])))

    def choose_best_action(self, state):
        action = np.argmax(self.q_table[state])
        return action

    def choose_action(self, state):
        if random.uniform(0, 1) < self.eps:
            action = self.env.action_space.sample()  # Explore action space using greedypolicy
        else:
            action = self.choose_best_action(state)  # Exploit learned values, take the best
        return action

    def update(self, state, action, new_state, reward, done):
        new_q_value = np.max(self.q_table[new_state])
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + (self.gamma * new_q_value * (1 - done)))
        self.q_table[state][action] = updated_q_value
