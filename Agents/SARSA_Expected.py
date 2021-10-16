import numpy as np
from Agents.SARSA_agent  import SARSA_Agent

class SARSA_Expected_Agent(SARSA_Agent):

    def __init__(self, envs,name,type, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(SARSA_Expected_Agent,self).__init__(envs, name,type,num_episodes, gamma, eps, lr)

    def update(self, state, action, new_state, reward, done):
        new_q_value = np.mean(self.q_table[new_state][:])
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * new_q_value)
        self.q_table[state][action] = updated_q_value