from random import random
import numpy as np
from agents.generic_agents import Agent
from agents.SARSA_agent  import SARSA_Agent

class SARSA_Expected_Agent(SARSA_Agent):

    def __init__(self, env, num_episodes, gamma, epsilon, alpha):
        super(SARSA_Expected_Agent,self).__init__(env, num_episodes, gamma, epsilon, alpha)

    def update(self, state, action, new_state, reward,done,current_episode):
        new_q_value = np.mean(self.q_table[new_state][:])
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * new_q_value)
        self.q_table[state][action] = updated_q_value