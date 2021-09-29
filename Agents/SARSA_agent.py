from random import random
import numpy as np
from Agents.generic_agents import Agent
from Agents.q_learning_agent import QAgent


class SARSA_Agent(QAgent):

    def __init__(self, env, name,type, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(SARSA_Agent, self).__init__(env, name,type,num_episodes, gamma, eps, lr)

    def update(self, state, action, new_state, reward, done):
        new_action = self.choose_action(new_state)
        new_q_value = self.q_table[new_state][new_action]
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * new_q_value)
        self.q_table[state][action] = updated_q_value
