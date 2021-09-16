from Agents.generic_agents import Agent
import sys
from os import path
sys.path.append(path.abspath('../Documents/storage-balancing-env'))
from data_processing.database import load_db



class TimeCorAgent(Agent):
    def __init__(self, env, num_episodes, gamma, eps=0.1, lr=0.1, anneal_lr_param=1., anneal_epsilon_param=1., threshold_lr_anneal=100., evaluate_every_n_episodes=200):
        super(TimeCorAgent, self).__init__(env, num_episodes, gamma, eps, lr, anneal_lr_param,anneal_epsilon_param)
        self.database = load_db( start_date="1/1/2015 0:00", end_date="1/2/2015 0:00", Dt="15min",imbalance_file="data/Imbalance_2015_16.csv")
        self.pricecounter=0
        self.cur_val=self.database.values[self.pricecounter,0]
        self.prv_val=self.database.values[self.pricecounter,0]


    def choose_action(self, state):
        if self.cur_val>self.prv_val:
            action=-1
        else:
            action=1
        if self.pricecounter==self.database.max_steps-1:self.pricecounter=0
        #TODO fix pricecounter and updates need episode counter
        self.prv_val=self.database.values[self.pricecounter,0]
        self.pricecounter+=1
        self.cur_val=self.database.values[self.pricecounter,0]
        return action

    def choose_best_action(self, state):
        action=self.choose_action(state)
        return action
