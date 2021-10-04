import gym
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import RL_storage_env  #### Important to keep here
import numpy as np
from collections import defaultdict

from Agents.q_learning_agent import QAgent
from Agents.SARSA_agent import SARSA_Agent
from Agents.SARSA_Expected import SARSA_Expected_Agent
from Agents.N_Steps import SarsaNStepsAgent, QNStepsAgent
from Agents.Double_Q_agent import DoubleQ_Agent
from Agents.LinearFunctionApproximation import Linear
from Agents.reinforce import ReinforceAgent
from Agents.random_agent import RandomAgent
from Agents.threshold_agent import ThresholdAgent
# from Agents.LinearFunctionApproximation_v2 import LFA_agent
from functions.wrappers import StateDiscretize, ActionDiscretize


def make_envs(my_dict):
    final_envs = dict()
    for env_name, vals in my_dict["Environments"].items():
        envir = create_parameters(env_name, vals)
        wrapped_env = create_wrappers(envir, vals)
        final_envs[env_name] = wrapped_env
    return final_envs


def create_parameters(env_name, vals):
    envir = list()
    for params in ["train_parameters", "test_parameters"]:
        env = gym.make(env_name, **vals[params])
        envir.append(env)
    return envir


def create_wrappers(envir, vals):
    wrapped_env = list()
    for env in envir:
        for wrapper, args in vals["wrappers"].items():
            env = eval(wrapper)(env, **args)
        wrapped_env.append(env)
    return wrapped_env


def make_agents(env, my_dict):
    final_ag = list()
    for ag_name, vals in my_dict["Agents"].items():
        ag = eval(vals["type"])(env, ag_name, **vals)
        final_ag.append(ag)
    return final_ag


def plot_performance(envs_agents, exp_path):
    for env, agents in envs_agents.items():
        plt.figure()
        plt.title("Performance")
        plt.suptitle("Cumulative rewards on " + env + "for training", fontsize='small')
        for agent in agents:
            plt.plot(agent.total_train_rewards, label=agent.name)
        plt.legend()
        plt.savefig('%s/Cumulative rewards for training.png' % (exp_path))
        plt.figure()
        plt.title("Performance")
        plt.suptitle("Cumulative rewards" + env + "for testing", fontsize='small')
        for agent in agents:
            plt.plot(agent.total_test_rewards, label=agent.name)
        plt.legend()
        plt.savefig('%s/Cumulative rewards for testing.png' % (exp_path))


def save_agent(agent, exp_path):
    with open(exp_path + '/' + agent.name + '.pkl', 'wb') as outp:
        pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)


def custom_hook(obj):
    # Identify dictionary with duplicate keys...
    # If found create a separate dict with single key and val and as list.
    if len(obj) > 1 and len(set(i for i, j in obj)) == 1:
        data_dict = defaultdict(list)
        for i, j in obj:
            data_dict[i].append(j)
        return dict(data_dict)
    return dict(obj)
