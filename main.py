import argparse
import time
import json
import os
from datetime import datetime
import gym
import RL_storage_env

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

from functions.utils import make_envs, make_agents, save_agent, plot_performance, custom_hook
from functions.wrappers import StateDiscretize, ActionDiscretize
from shutil import copy


def get_config_file():
    parser = argparse.ArgumentParser(description='Run RL agents.')
    parser.add_argument('--config_file', help='Path to config file')
    args = parser.parse_args()

    config_file = args.config_file
    return config_file


def get_exp_dir():
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)
    return exp_path


def load_input_data(config_file):
    with open(config_file, 'rb') as f:
        data = json.load(f)
    return data


def create_envs_agents_combinations(data):
    envs = make_envs(data)

    envs_agents = dict()
    for env_name, env in envs.items():
        envs_agents[env_name] = make_agents(env, data)
    return envs_agents


def run(env_agents):
    for env, agents in envs_agents.items():
        for agent in agents:
            agent.train()
            agent.plot(exp_path)
            save_agent(agent, exp_path)


if __name__ == '__main__':
    config_file = get_config_file()

    exp_path = get_exp_dir()

    copy(config_file, exp_path)

    data = load_input_data(config_file)

    envs_agents = create_envs_agents_combinations(data)

    run(envs_agents)

    plot_performance(envs_agents, exp_path)
