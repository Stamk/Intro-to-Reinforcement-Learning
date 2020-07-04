from random import random
import numpy as np
from agents.generic_agents import Agent
from agents.q_learning_agent import QAgent

class SARSA_Agent(QAgent):

    def __init__(self, env, num_episodes, gamma, epsilon, alpha):
        super(SARSA_Agent,self).__init__(env, num_episodes, gamma,epsilon, alpha)

    def update(self, state, action, new_state, reward,done,current_episode,episode_length):
        new_action = self.choose_action(new_state)
        new_q_value = self.q_table[new_state][new_action]
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * new_q_value)
        self.q_table[state][action] = updated_q_value