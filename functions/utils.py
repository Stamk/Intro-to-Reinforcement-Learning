import RL_storage_env  #### Important to keep here
import gym
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy
from datetime import datetime
import os
import json

from Agents.q_learning_agent import QAgent
from Agents.SARSA_agent import SARSA_Agent
from Agents.SARSA_Expected import SARSA_Expected_Agent
from Agents.N_Steps import SarsaNStepsAgent, QNStepsAgent
from Agents.Double_Q_agent import DoubleQ_Agent
from Agents.LinearFunctionApproximation import Linear
from Agents.reinforce import ReinforceAgent
from Agents.random_agent import RandomAgent
from Agents.threshold_agent import ThresholdAgent
from Agents.LinearFunctionApproximation_v2 import LFA_agent
from Agents.deep_q_network import Deep_Q_Agent
from functions.wrappers import StateDiscretize, ActionDiscretize, PricesAddition
from gym.wrappers.frame_stack import FrameStack,LazyFrames
from gym.wrappers.flatten_observation import FlattenObservation



def get_config_file():
    """
    Constructs the parser for the configuration file

    Return:
        :return: the configuration file
    """
    parser = argparse.ArgumentParser(description='Run RL agents.')
    parser.add_argument('--config_file', required=True, help='Path to config file')
    args = parser.parse_args()
    config_file = args.config_file
    return config_file


def load_input_data(config_file):
    """
    Reads the data from the json file in order to be processed
    Args:
        :param config_file: path to json file with respective configurations

    Return:
        :return: raw data being loaded from the file
    """
    with open(config_file, 'rb') as f:
        data = json.load(f)
    return data


def make_envs(my_dict):
    """
    Constructs the environments according to the parameters and wrappers given
    Args:
        :param my_dict: dictionary with the environments

    Return:
        :return: the final environments
    """
    final_envs = dict()
    for env_name, vals in my_dict["Environments"].items():
        envir = create_parameters(env_name, vals)
        wrapped_env = create_wrappers(envir, vals)
        final_envs[env_name] = wrapped_env
    return final_envs


def create_parameters(env_name, vals):
    """
    Makes the new environment with the parameters given

    Args:
        :param env_name: environment name
        :param vals: parameters for the environment

    Return:
        :return: the new environment according to parameters
    """
    envir = list()
    for params in ["train_parameters", "test_parameters"]:
        env = gym.make(env_name, **vals[params])
        envir.append(env)
    return envir


def create_wrappers(envir, vals):
    """
    Constructs the final environments by wrapping them with the wrappers given
    Args:
        :param envir: environment
        :param vals: wrappers

    Return:
        :return: a list with the final environments passed through wrappers
    """
    wrapped_env = list()
    for env in envir:
        for wrapper, args in vals["wrappers"].items():
            env = eval(wrapper)(env, **args)
        wrapped_env.append(env)
    return wrapped_env


def make_agents(env, my_dict):
    """
    It makes agents according to the parameters given
    Args:
        :param env: environment of the agent
        :param my_dict: dictionary with Agents

    Return:
        :return: a list with the final agents
    """
    final_ag = list()
    for ag_name, vals in my_dict["Agents"].items():
        ag = eval(vals["type"])(env, ag_name, **vals)
        final_ag.append(ag)
    return final_ag


def create_envs_agents_combinations(data):
    """
    It creates a dictionary with all the agents and their environments

    Args:
        :param data: given data from json file
    Return:
        :return: dictionary
    """
    envs = make_envs(data)
    envs_agents = dict()
    for env_name, env in envs.items():
        envs_agents[env_name] = make_agents(env, data)
    return envs_agents


def get_exp_dir():
    """
    Creates and returns a working dictionary according to current date and time in order to save the results of the experiment

    Return:
        :return: returns the path created
    """
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)
    return exp_path


def save_agent(agent, exp_path):
    """
    Stores as a pickle the trained agent after the experiment ends

    Args:
        :param agent: the class of trained agent
        :param exp_path: path to save the results

    Return:
        :return: .pkl file of the trained agent
    """
    with open(exp_path + '/' + agent.name + '.pkl', 'wb') as outp:
        pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)


def filter_list(res_list, alpha=0.9):
    """
    It makes a list of values being smoother combined. It multiplies and combines the nearests elements by a factor alpha

    Args:
        :param res_list: input list
        :param alpha: factor

    Return:
        :return: the filtered list
    """
    c = 0.
    filtered_list = [c]
    for it in res_list:
        c = alpha * c + (1. - alpha) * it
        filtered_list.append(c)

    return filtered_list


def plot_performance(envs_agents, exp_path):
    """
    Plots all the scores that made each agent on evaluation in training and testing environment
    Args:
        :param envs_agents: dictionary with environments and agents from json file
        :param exp_path: path to save the results
    Returns:
        :return: saves the figures in the directory
    """
    for env, agents in envs_agents.items():
        for param in ["train", "test"]:
            plt.figure()
            plt.title("Performance")
            plt.suptitle("Cumulative rewards on evaluation in " + env + " for " + param, fontsize='small')
            for agent in agents:
                res_list = getattr(agent, "total_" + param + "_rewards")
                plt.plot(filter_list(res_list, alpha=0.9), label=agent.name)
            plt.legend()
            plt.savefig(exp_path + "/Cumulative rewards on evaluation in " + env + " for " + param + ".png")


def plot_evaluation_traces(agent, exp_path):
    """
    Saves the plots of all states,actions and rewards traces while training and testing

    Args:
        :param agent: the class of trained agent
        :param exp_path: path to save the results

    Return:
        :return: saves the figures in the directory
    """
    for param in ["train", "test"]:
        plt.figure()
        plt.title(agent.name + param)
        states = getattr(agent, param + "_states")
        ax1 = plt.subplot(311)
        ax1.set_title(param + " states")
        ax1.plot((numpy.reshape(states, (states.__len__(), states[0].shape[0])))[:, 0])
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title(param + " actions")
        actions = getattr(agent, param + "_actions")
        ax2.plot(actions)
        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title(param + " rewards")
        rewards = getattr(agent, param + "_rewards")
        ax3.plot(rewards)
        env = getattr(agent, param + "_env")
        plt.savefig('%s/%s agent of type %s on %s for %d episodes with learning rate %s and gamma %s for %s.png' % (
            exp_path, agent.name, agent.type, env.spec.id, agent.num_episodes, agent.lr, agent.gamma, param))


def run(envs_agents, exp_path):
    """
    Executes the trainig process for the agents in the environments
    Args:
        :param envs_agents: dictionary with environments and agents from json file
        :param exp_path: path to save the results
    """
    for env, agents in envs_agents.items():
        for agent in agents:
            agent.train()
            plot_evaluation_traces(agent,exp_path)
            save_agent(agent, exp_path)
