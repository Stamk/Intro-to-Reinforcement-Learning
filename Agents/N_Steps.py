import numpy as np
from copy import deepcopy
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
        if not done: #Initialize and store So, not terminal
                     #Select and store an Action Ao accordinf to policy
            self.states.append(state)
            self.actions.append(action)
        t=0
        while True:
            if t<self.T:
                 #take action At according to policy
                 #observe and store the next reward as R(t+1) and the next state as S(t+1)
                 if t==0:
                     self.store_transitions(new_state, reward,done)
                 else:
                     new_state, reward, done, info = deepcopy(self.env.step(action))
                     self.store_transitions(new_state, reward,done)

                 if done:
                     # if S(t+1) is terminal, then T=t+1
                     self.T=t+1
                 else:
                     #select and store an action A(t+1) according to policy
                     action = deepcopy(self.choose_action(new_state))
                     self.actions.append(action)
            tau=t-self.N+1
            G = 0
            if tau>=0:
                for i in range(tau+1, min(tau + self.N, self.T)):
                    G += np.power(self.gamma, i - tau - 1) * self.rewards[i]
                if tau + self.N < self.T:
                    state_tau_plus_n=self.states[tau+self.N-1]
                    action_tau_plus_n=self.actions[tau+self.N-1]
                    G+=np.power(self.gamma, self.N) * self.q_table[state_tau_plus_n][action_tau_plus_n] # TODO fix this
                final_state = new_state
                final_action = self.actions[tau + self.N-1]
                G += np.power(self.gamma, self.N) * self.value_estimate(final_state, final_action)
                init_state = self.states[tau]
                init_action = self.actions[tau]
                self.q_table[init_state][init_action] += self.lr * (G - self.q_table[init_state][init_action])
            t=t+1
            if (tau==self.T-1):break

    def store_transitions(self, state, reward, done):
        self.states.append(state)
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
