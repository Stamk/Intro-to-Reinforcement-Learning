from Agents.generic_agents import Agent


class RandomAgent(Agent):
    def __init__(self, envs,name,type, num_episodes,gamma):
        super(RandomAgent, self).__init__(envs,name,type, num_episodes,gamma)

    def choose_action(self, state,env):
        return env.action_space.sample().item()

    def choose_best_action(self, state,env):
        return env.action_space.sample().item()
