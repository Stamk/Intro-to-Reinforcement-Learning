import random
import numpy as np
import copy
from agents.generic_agents import Agent
from copy import deepcopy

class Linear(Agent):
        def __init__(self, env, num_episodes, gamma, epsilon, alpha):
            super(Linear, self).__init__(env, num_episodes, gamma)
            #initialize weight and bias
            self.w_shape = tuple(self.env.observation_space.nvec) + (self.env.action_space.n,)
            self.w = np.ones([4])
            self.x=np.ones([4,1])
            self.bias = 0.0
            self.epsilon = epsilon
            self.gamma = gamma
            self.alpha = alpha
            self.h=1

        def choose_action(self, state):
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore action space using greedypolicy
            else:
                action = self.choose_best_action(state)  # Exploit learned values, take the best
            return action
        #update parameters
        def choose_best_action(self, state):
            return self.env.action_space.sample()

        def update(self, state, action, new_state, reward, done, current_episode, episode_length):
            lqf_s_a = self.linear_calc(state, action) #actions given state s
            new_action = self.choose_action(new_state)
            lqf_s_a_next = self.linear_calc(new_state, new_action)
            delta=reward+self.gamma*lqf_s_a_next-lqf_s_a
            print(lqf_s_a,lqf_s_a_next)
            for i in range(0,4):
             self.w[i]=self.w[i]+self.h*delta*state[i]
#            TD_error = self.alpha * (lqf_s_a - (reward + self.gamma * lqf_s_a_next))
#            if TD_error != 0.0:
#                for state,v in states: #i = row, a = column, v = value of the state
#                    self.w[state, action] = self.w[state, action] - TD_error * v
#                self.bias = self.bias - TD_error
            return (False or done)

        #return q(s, a; w) for a given state s and action a
        def linear_calc(self, state, action):
            product = 0.0
#            for state,v in states:
#                product += v * self.w[state, action]
            product=np.dot(self.w,state)
            return product + self.bias


'''        def update_weights(self, weights, g):
            for i in range(len(weights)):
                for param in weights[i].keys():
                    weights[i][param] += self.step_size * g[i][param]
'''

