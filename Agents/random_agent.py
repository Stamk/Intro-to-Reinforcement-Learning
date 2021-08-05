from Agents.generic_agents import Agent


class RandomAgent(Agent):
    def __init__(self, env, num_episodes, gamma, lr=0.1, eps=0.1, anneal_lr_param=1., anneal_epsilon_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(RandomAgent, self).__init__(env, num_episodes, gamma, lr, eps, anneal_lr_param,
                                          anneal_epsilon_param,
                                          threshold_lr_anneal, evaluate_every_n_episodes)

    def train(self):
        pass

    def choose_best_action(self, state):
        return self.env.action_space.sample().item()
