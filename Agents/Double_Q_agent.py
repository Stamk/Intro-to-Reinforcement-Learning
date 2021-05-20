from random import random
import numpy as np
from Agents.generic_agents import Agent
from Agents.q_learning_agent import QAgent

class DQ_Agent(QAgent):

    def __init__(self, env, num_episodes, gamma, epsilon, alpha):
        super(DQ_Agent,self).__init__(env, num_episodes, gamma,epsilon, alpha)
        self.q_table_B = np.zeros(self.q_table_shape)

    def update(self, state, action, new_state, reward,done,current_episode,episode_length):
        p = np.random.random()
        if (p < .5):
            action_a= np.argmax(self.q_table[new_state])
            self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table_B[new_state][action_a] - self.q_table[state][action])
        else:
            action_b= np.argmax(self.q_table_B[new_state])
            self.q_table_B[state][action] = self.q_table_B[state][action] + self.alpha * (reward + self.gamma * self.q_table[new_state][action_b] - self.q_table_B[state][action])