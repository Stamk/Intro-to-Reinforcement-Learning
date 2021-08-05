from Agents.generic_agents import Agent


class ThresholdAgent(Agent):
    def __init__(self, env, num_episodes, gamma, threshold=[20, 50], lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(ThresholdAgent, self).__init__(env, num_episodes, gamma, lr, eps, anneal_lr_param,
                                          anneal_epsilon_param,
                                          threshold_lr_anneal, evaluate_every_n_episodes)
        self.threshold = threshold

    def train(self):
        pass

    def choose_best_action(self, state):
        action = 0
        if state[-1] > self.threshold[1]:
            action = -1
        elif state[-1] < self.threshold[0]:
            action = 1
        return action
