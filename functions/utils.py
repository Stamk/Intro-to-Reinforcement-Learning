import gym
import matplotlib.pyplot as plt


def make_envs(my_dict):
    final_envs = list()
    for env_name, vals in my_dict["Environments"].items():
        env = gym.make(env_name, **vals)
        final_envs.append(env)
    return final_envs


def make_agents(env, my_dict):
    final_ag = list()
    for ag_name, vals in my_dict["Agents"].items():
        ag = eval(ag_name)(env, **vals)
        final_ag.append(ag)
    return final_ag


def plot_performance(envs_agents, exp_path):
    for env, agents in envs_agents.items():
        plt.figure()
        plt.title(env.spec.id)
        for agent in agents:
            plt.plot(agent.results, label=agent.__class__.__name__)
        plt.legend()
        plt.savefig('%s/%s.png' % (exp_path, env.unwrapped.__class__.__name__))
