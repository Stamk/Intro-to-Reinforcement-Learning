import numpy as np
from copy import deepcopy
from Agents.q_learning_agent import QAgent


class NStepsAgent(QAgent):
    def __init__(self, envs, name,type,num_episodes, gamma, number_steps=1, lr=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(NStepsAgent, self).__init__(envs, name,type,num_episodes, gamma, lr, anneal_lr_param,
                                          anneal_epsilon_param,
                                          threshold_lr_anneal, evaluate_every_n_episodes)
        self.N = number_steps
        self.init_buffers()

    def init_buffers(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = [0]


    def simulate(self, policy, train_flag=False,eval_flag=False):
        """
        :param policy:
        :param train_flag:
        :return:
        """
        done = False
        cum_reward = 0.
        reward=0.
        env = self.test_env if eval_flag else self.train_env
        state = env.reset()
        t=0
        self.T=np.inf
        while not done:
            tau = t - self.N + 1
            action = policy(state,env)
            self.store_buffer_transitions(state,action,reward,done)
            new_state, reward, done, info = env.step(action)
            if done: self.T=t+1
            if not train_flag: self.store_transitions(state,action,reward)
            if train_flag and tau>=0 : self.update_vol2(state, action, new_state, reward, done, tau=tau) #TODO make simulate,update abstact for all agents?
            cum_reward += reward
            state = deepcopy(new_state)
            t+=1
        return cum_reward


    def update_vol2(self, state, action, new_state, reward, done,tau): #TODO make simulate,update abstact for all agents?
        if not done: #Initialize and store So, not terminal
                     #Select and store an Action Ao accordinf to policy
            self.store_buffer_transitions(state,action,reward,done)
        else:
            final_state = deepcopy(new_state)
        G = 0
        for i in range(tau+1, min(tau + self.N, self.T)):
            G += np.power(self.gamma, i - tau - 1) * self.buffer_rewards[i]
        if tau + self.N < self.T:
            state_tau_plus_n=self.buffer_states[tau+self.N]
            action_tau_plus_n=self.buffer_actions[tau+self.N]
            G+=np.power(self.gamma, self.N) * self.q_table[state_tau_plus_n][action_tau_plus_n]
        init_state = self.buffer_states[tau]
        init_action = self.buffer_actions[tau]
        self.q_table[init_state][init_action] += self.lr * (G - self.value_estimate(init_state,init_action))
        if (tau==self.T-1) or done:
         self.init_buffers()

    def store_buffer_transitions(self, state, action, reward, done):
        self.buffer_states.append(state)
        self.buffer_rewards.append(reward)
        self.buffer_actions.append(action)
        if done:
            print("End at state {} | number of states {}".format(state, len(self.buffer_states)))
            #self.T = len(self.N_states)
            self.init_buffers()

    def value_estimate(self, s, a):
        return NotImplementedError


class SarsaNStepsAgent(NStepsAgent):
    def __init__(self, envs,name,type, num_episodes, gamma, number_steps=1, lr=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(SarsaNStepsAgent, self).__init__(envs, name,type,num_episodes, gamma, number_steps, lr, anneal_lr_param,
                                               anneal_epsilon_param,
                                               threshold_lr_anneal, evaluate_every_n_episodes)

    def value_estimate(self, s, a):
        return self.q_table[s][a]


class QNStepsAgent(NStepsAgent):
    def __init__(self, envs,name,type, num_episodes, gamma, number_steps=1, lr=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(QNStepsAgent, self).__init__(envs, name,type,num_episodes, gamma, number_steps, lr, anneal_lr_param,
                                           anneal_epsilon_param,
                                           threshold_lr_anneal, evaluate_every_n_episodes)

    def value_estimate(self, s, a):
        return np.max(self.q_table[s])
