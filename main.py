import os
from datetime import datetime
from time import strftime

import gym
from agents.q_learning_agent import QAgent
from agents.SARSA_agent  import SARSA_Agent
from agents.SARSA_Expected import SARSA_Expected_Agent
from agents.N_Steps import Nsteps_agent
from agents.Double_Q_agent import DQ_Agent
from agents.Value_iteration import ValueIteraion
from environments.generic_env import Discretize

if __name__ == '__main__':

     num_episodes, gamma, epsilon, alpha, N = 7000, 0.99, 0.05, 1, 5
     exp_path = "results/%s"%(datetime.now().strftime("%Y_%m_%d_%H%M%S"))
     os.makedirs(exp_path)
  #   env1 = gym.make('CartPole-v0')
  #   env1 = Discretize(env1,stepsize=15,threshold=10)
  #   ag11 = QAgent(env1, num_episodes, gamma, epsilon, alpha)
  #   ag12 = SARSA_Agent(env1, num_episodes, gamma, epsilon, alpha)
  #   ag13 = SARSA_Expected_Agent(env1, num_episodes, gamma, epsilon, alpha)

     env2 = gym.make('MountainCar-v0')
     env2 = Discretize(env2,stepsize=3)
#     ag21 = QAgent(env2, num_episodes, gamma, epsilon, alpha)
#     ag22 = SARSA_Agent(env2, num_episodes, gamma, epsilon, alpha)
#     ag23 = SARSA_Expected_Agent(env2, num_episodes, gamma, epsilon, alpha)
#     ag25 = DQ_Agent(env2, num_episodes, gamma, epsilon, alpha)
     ag26 = ValueIteraion(env2,num_episodes, gamma)
     ag26.Value()


#     ag4 = Nsteps_agent(env2, num_episodes, gamma, epsilon, alpha,N)
#     ag4.train()


#     ag21.train()
#     ag21.evaluate()
 #    ag21.plot(exp_path)

 #    ag23.train()
 #    ag23.evaluate()
 #    ag23.plot(exp_path)
    #print(ag21.q_table)