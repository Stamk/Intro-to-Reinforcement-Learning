
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
from Agents.LinearFunctionApproximation_v2 import LFA_agent

from functions.utils import make_envs, make_agents
from functions.wrappers import StateDiscretize,ActionDiscretize

if __name__ == '__main__':
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)

    config_file = 'data/storage_agent.json'

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
        # agent.evaluate()
    print("I m here")
