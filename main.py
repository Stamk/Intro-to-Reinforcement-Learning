import time
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

from functions.utils import *
from functions.wrappers import StateDiscretize, ActionDiscretize
from shutil import copy

if __name__ == '__main__':
    config_file = get_config_file()

    exp_path = get_exp_dir()

    copy(config_file, exp_path)

    data = load_input_data(config_file)

    envs_agents = create_envs_agents_combinations(data)

    run(envs_agents, exp_path)

    plot_performance(envs_agents, exp_path)
