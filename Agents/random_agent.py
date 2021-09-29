from Agents.generic_agents import Agent


class RandomAgent(Agent):
    def __init__(self, env,name,type, num_episodes, gamma, lr=0.1, anneal_lr_param=1.,
                 threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(RandomAgent, self).__init__(env,name,type, num_episodes, gamma, lr, anneal_lr_param,
                                          threshold_lr_anneal, evaluate_every_n_episodes)

    def choose_action(self, state):
        return self.env.action_space.sample().item()

    def choose_best_action(self, state):
        return self.env.action_space.sample().item()
