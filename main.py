import json
import os
from datetime import datetime
from time import strftime
import matplotlib.pyplot as plt

import gym

from Agents.q_learning_agent import QAgent
from Agents.SARSA_agent import SARSA_Agent
from Agents.SARSA_Expected import SARSA_Expected_Agent
from Agents.N_Steps import Nsteps_agent
from Agents.Double_Q_agent import DQ_Agent
from Agents.LinearFunctionApproximation import Linear
from Agents.reinforce import ReinforceAgent
from Environments.generic_env import Discretize


def make_envs(my_dict):
    final_envs = list()
    for env_name, vals in my_dict["Environments"].items():
        env = gym.make(env_name)
        for wrapper, args in vals["wrappers"].items():
            env = eval(wrapper)(env, **args)
        final_envs.append(env)
    return final_envs


def make_agents(env, my_dict):
    final_ag = list()
    for ag_name, vals in my_dict["Agents"].items():
        ag = eval(ag_name)(env, **vals)
        final_ag.append(ag)
    return final_ag


def plot_performance(envs_agents):
    for env, agents in envs_agents.items():
        plt.figure()
        plt.title(env.spec.id)
        for agent in agents:
            plt.plot(agent.results, label=agent.__class__.__name__)
        plt.legend()
        plt.savefig('%s/%s.png' % (exp_path, env.unwrapped.__class__.__name__))


if __name__ == '__main__':
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)

    config_file = 'data/My_dict.json'

    with open(config_file, 'rb') as f:
        my_dict = json.load(f)

    envs = make_envs(my_dict)
    envs_agents = dict()
    for env in envs:
        envs_agents[env] = make_agents(env, my_dict)

    for env, agents in envs_agents.items():
        for agent in agents:
            agent.train()
            agent.save()

    plot_performance(envs_agents)
    print("I m here")
