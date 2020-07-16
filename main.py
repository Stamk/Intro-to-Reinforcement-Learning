import json
import os
from datetime import datetime
from time import strftime
import matplotlib.pyplot as plt

import gym

from agents.q_learning_agent import QAgent
from agents.SARSA_agent import SARSA_Agent
from agents.SARSA_Expected import SARSA_Expected_Agent
from agents.N_Steps import Nsteps_agent
from agents.Double_Q_agent import DQ_Agent
from agents.Value_iteration import ValueIteraion
from agents.LinearFunctionApproximation import Linear
from environments.generic_env import Discretize
 

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

    # num_episodes, gamma, epsilon, alpha, N = 10000, 0.99, 0.85, 1, 5
    # env1 = gym.make('CartPole-v0')
    # env1 = Discretize(env1, stepsizes=[10, 10, 20, 30])
    # ag11 = QAgent(env1, num_episodes, gamma, epsilon, alpha)
    # ag12 = DQ_Agent(env1, num_episodes, gamma, epsilon, alpha)
    # num_episodes, gamma, epsilon, alpha, N = 10000, 0.99, 0.85, 1, 3
    # ag13 = SARSA_Agent(env1, num_episodes, gamma, epsilon, alpha)
    # ag14 = SARSA_Expected_Agent(env1, num_episodes, gamma, epsilon, alpha)
    # ag15 = Nsteps_agent(env1, num_episodes, gamma, epsilon, alpha, N)
    #
    # num_episodes, gamma, epsilon, alpha, N = 10000, 0.99, 0.25, 1, 3
    # env2 = gym.make('MountainCar-v0')
    # env2 = Discretize(env2, stepsizes=[40, 40])
    # ag21 = QAgent(env2, num_episodes, gamma, epsilon, alpha)
    # ag22 = DQ_Agent(env2, num_episodes, gamma, epsilon, alpha)
    # ag23 = SARSA_Agent(env2, num_episodes, gamma, epsilon, alpha)
    # ag24 = SARSA_Expected_Agent(env2, num_episodes, gamma, epsilon, alpha)
    # ag25 = Nsteps_agent(env2, num_episodes, gamma, epsilon, alpha, N)

    # ag11.train()
    # ag11.plot(exp_path)
    # ag12.train()
    # ag12.plot(exp_path)
    # ag13.train()
    # ag13.plot(exp_path)
    # ag14.train()
    # ag14.plot(exp_path)
    # ag15.train()
    # ag15.plot(exp_path)

    # ag21.train()
    # ag21.plot(exp_path)
    # ag22.train()
    # ag22.plot(exp_path)
    # ag23.train()
    # ag23.plot(exp_path)
    # ag24.train()
    # ag24.plot(exp_path)
    # ag25.train()
    # ag25.plot(exp_path)
