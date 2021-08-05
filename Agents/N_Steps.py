import numpy as np
from Agents.q_learning_agent import QAgent


class NStepsAgent(QAgent):
    def __init__(self, env, num_episodes, gamma, number_steps=5, lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(NStepsAgent, self).__init__(env, num_episodes, gamma, lr, eps, anneal_lr_param,
                                          anneal_epsilon_param,
                                          threshold_lr_anneal, evaluate_every_n_episodes)
        self.N = number_steps

        self.states = None
        self.actions = None
        self.rewards = None
        self.init_buffers()
        self.T = np.inf

    def init_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def update(self, state, action, new_state, reward, done):
        self.store_transitions(state, action, reward, done)
        episode_length = len(self.states)
        tau = episode_length - self.N + 1

        if tau >= 0:
            init_state = self.states[tau]
            init_action = self.actions[tau]
            G = 0

            for i in range(tau, min(tau + self.N-1, self.T-1)):
                G += np.power(self.gamma, i - tau - 1) * self.rewards[i]

            if tau + self.N-1 < self.T: # TODO fix this
                final_state = new_state
                final_action = self.actions[tau + self.N-1]
                G += np.power(self.gamma, self.N) * self.value_estimate(final_state, final_action)

            self.q_table[init_state][init_action] += self.lr * (
                    G - self.q_table[init_state][init_action])

    def store_transitions(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if done:
            print("End at state {} | number of states {}".format(state, len(self.states)))
            self.T = len(self.states)
            self.init_buffers()

    def value_estimate(self, s, a):
        return NotImplementedError


class SarsaNStepsAgent(NStepsAgent):
    def __init__(self, env, num_episodes, gamma, number_steps=5, lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(SarsaNStepsAgent, self).__init__(env, num_episodes, gamma, number_steps, lr, eps, anneal_lr_param,
                                               anneal_epsilon_param,
                                               threshold_lr_anneal, evaluate_every_n_episodes)

    def value_estimate(self, s, a):
        return self.q_table[s][a]


class QNStepsAgent(NStepsAgent):
    def __init__(self, env, num_episodes, gamma, number_steps=5, lr=0.1, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(QNStepsAgent, self).__init__(env, num_episodes, gamma, number_steps, lr, eps, anneal_lr_param,
                                           anneal_epsilon_param,
                                           threshold_lr_anneal, evaluate_every_n_episodes)

    def value_estimate(self, s, a):
        return np.max(self.q_table[s])
