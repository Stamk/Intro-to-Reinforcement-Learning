import gym
import matplotlib.pyplot as plt
import RL_storage_env  #### Important to keep here
from Agents.q_learning_agent import QAgent
from Agents.SARSA_agent import SARSA_Agent
from Agents.SARSA_Expected import SARSA_Expected_Agent
from Agents.N_Steps import SarsaNStepsAgent, QNStepsAgent
from Agents.Double_Q_agent import DoubleQ_Agent
from Agents.LinearFunctionApproximation import Linear
from Agents.reinforce import ReinforceAgent
from Agents.random_agent import RandomAgent
from Agents.threshold_agent import ThresholdAgent
from Agents.TimeCorrelation import TimeCorAgent
from Agents.LinearFunctionApproximation_v2 import LFA_agent
from functions.wrappers import StateDiscretize,ActionDiscretize

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



def plot_performance(envs_agents, exp_path):
    for env, agents in envs_agents.items():
        plt.figure()
        plt.title(env.spec.id)
        for agent in agents:
            plt.plot(agent.results, label=agent.__class__.__name__)
        plt.legend()
        plt.savefig('%s/%s.png' % (exp_path, env.unwrapped.__class__.__name__))
