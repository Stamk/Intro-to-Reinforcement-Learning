import random
import numpy as np
from agents.generic_agents import Agent


class QAgent(Agent):

    def __init__(self, env, num_episodes, gamma, epsilon, alpha):
        super(QAgent,self).__init__(env, num_episodes, gamma)
        self.epsilon = epsilon
        self.alpha = alpha
        States= self.env.observation_nvec()
        #q_table_shape = (States,) +self.env.observation_space.shape +(self.env.action_space.n,)
        self.q_table_shape = States + (self.env.action_space.n,)
        self.q_table = np.zeros(self.q_table_shape)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore action space using greedypolicy
        else:
            action = np.argmax(self.q_table[state])  # Exploit learned values, take the best
        #self.linear_decay(self.epsilon)
        return action

    def choose_best_action(self, state):
        action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, new_state, reward,done,current_episode,episode_length):
        if current_episode%100==0:
            self.linear_decay(self.alpha)
            self.linear_decay(self.gamma)
            self.linear_decay(self.epsilon)
        new_q_value = np.max(self.q_table[new_state])
        old_q_value = self.q_table[state][action]
        updated_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * new_q_value)
        self.q_table[state][action] = updated_q_value

    @staticmethod
    def linear_decay(lr):
         lr=lr * 0.99

    @staticmethod
    def exp_decay(lr):
        return lr * 0.99

    def my_decay(self ):
        pass
      #  return self.counter*