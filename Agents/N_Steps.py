import numpy as np
import random
from agents.generic_agents import Agent
from agents.q_learning_agent import QAgent

class Nsteps_agent(QAgent):
    def __init__(self, env, num_episodes, gamma, epsilon, alpha, N):
         super(Nsteps_agent,self).__init__(env, num_episodes, gamma,epsilon, alpha)
         self.N = N
         current_state = self.env.reset()
         action = self.choose_action(current_state)
         self.actions = [action]
         self.states = [current_state]
         self.rewards = [0]
         self.T = np.inf

       # self.states= dict{}

    def update(self, state, action, new_state, reward, done, current_episode,episode_length):
        self.store_transitions(state,action,done,episode_length)
        tau = episode_length - self.N + 1
        self.state_action = (self.states[tau + self.N], self.actions[tau + self.N])
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + self.N + 1, self.T + 1)):
                G += np.power(self.gamma, i - tau - 1) * self.rewards[i]
            if tau + self.N < self.T:
                state_action = (self.states[tau + self.N], self.actions[tau + self.N])
                G += np.power(self.gamma, self.N) * self.q_table[state_action[0]][state_action[1]]
            state_action = (self.states[tau], self.actions[tau])
            self.q_table[state_action[0]][state_action[1]] += self.alpha * (G - self.q_table[state_action[0]][state_action[1]])
        if tau == self.T - 1:
            done = True

    def store_transitions(self, state,reward,done,episode_length):
        self.states.append(state)
        self.rewards.append(reward)
        if done:
            print("End at state {} | number of states {}".format(state, len(states)))
            self.T = episode_length + 1
        else:
            action = self.choose_action(state)
            self.actions.append(action)

            """
                def train(self):
                 for _ in range(self.num_episodes):
                    current_state = self.env.reset()
                    t = 0
                    T = np.inf
                    action = self.choose_action(current_state)
                    actions = [action]
                    states = [current_state]
                    rewards = [0]
                    while True:
                        if t < T:
                            state, reward, done, info = self.env.step(action)
                            states.append(state)
                            rewards.append(reward)
                            if done:
                                print("End at state {} | number of states {}".format(state, len(states)))
                                T = t + 1
                            else:
                                action = self.choose_action(state)
                                actions.append(action)
                        # state tau being updated
                        tau = t - self.N + 1
                        if tau >= 0:
                            G = 0
                            for i in range(tau + 1, min(tau + self.N + 1, T + 1)):
                                G += np.power( self.gamma, i - tau - 1) * rewards[i]
                            if tau + self.N < T:
                                state_action = (states[tau + self.N], actions[tau + self.N])
                                G += np.power( self.gamma, self.N) * self.q_table[state_action[0]][state_action[1]]
                            state_action = (states[tau], actions[tau])
                            self.q_table[state_action[0]][state_action[1]] += self.alpha * (G - self.q_table[state_action[0]][state_action[1]])
                        if tau == T - 1:
                            break
                        t += 1
            """
