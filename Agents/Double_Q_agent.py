from copy import deepcopy

import numpy as np
from Agents.q_learning_agent import QAgent


class DoubleQ_Agent(QAgent):
    def __init__(self, env,name,type, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        """

        :param env:
        :param num_episodes:
        :param gamma:
        :param lr:
        :param eps:
        :param anneal_lr_param:
        :param anneal_epsilon_param:
        :param threshold_lr_anneal:
        :param evaluate_every_n_episodes:
        """
        super(DoubleQ_Agent, self).__init__(env, name,type,num_episodes, gamma, lr, eps, anneal_lr_param,
                                            anneal_epsilon_param,
                                            threshold_lr_anneal, evaluate_every_n_episodes)
        self.q_table_B = deepcopy(self.q_table)

    def update(self, state, action, new_state, reward, done):
        p = np.random.random()
        if p < .5:  # TODO fix hardcoded values
            action_a = np.argmax(self.q_table[new_state])
            self.q_table[state][action] = self.q_table[state][action] + self.lr * (
                    reward + self.gamma * self.q_table_B[new_state][action_a] - self.q_table[state][action])
        else:
            action_b = np.argmax(self.q_table_B[new_state])
            self.q_table_B[state][action] = self.q_table_B[state][action] + self.lr * (
                    reward + self.gamma * self.q_table[new_state][action_b] - self.q_table_B[state][action])

    def choose_best_action(self, state):
        action = np.argmax(self.q_table[state]+self.q_table_B[state])
        return action
