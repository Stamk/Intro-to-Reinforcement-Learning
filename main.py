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
from Agents.LinearFunctionApproximation_v2 import LFA_agent
from Agents.TimeCorrelation import TimeCorAgent


from functions.utils import make_envs, make_agents,save_agent,plot_performance
from functions.wrappers import StateDiscretize,ActionDiscretize

if __name__ == '__main__':
    exp_path = "results/%s" % (datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)
    t_0 = time.time()
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
            print("Training time: %.1f" % (time.time() - t_0))
            agent.plot(exp_path)
            agent.save(exp_path)
            save_agent(agent,'agent.pkl')
        # agent.evaluate()
    plot_performance(envs_agents)
    print("I m here")
