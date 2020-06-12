# Agent of player
import numpy as np
import random
import time
import copy

#constants for board setting
BOARD_ROWS = 3
BOARD_COLS = 4
NUM_OF_STATES = BOARD_ROWS * BOARD_COLS
NUM_OF_EPISODES = 50000
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START_STATE=(1,0)

# Hyperparameters
alpha = 0.7
gamma = 0.85
epsilon = 0.25


class Agent:

    def __init__(self):
        # initialize actions
        self.actions = ["up", "down", "left", "right"]
        #initialize reward table
        self.reward_table = np.zeros([NUM_OF_STATES, len(self.actions)])
        self.reward_table[2][3] = 1
        self.reward_table[6][3] = -1
        self.reward_table[11][0] = -1
        # initialize Q table
        self.q_table = np.zeros([NUM_OF_STATES, len(self.actions)])
        #initialize Cumulative Reward
        self.cum_reward=0

    def Q_values(self,str):
        # copy of Q-table for converge condition
        q_table_updated = np.zeros([NUM_OF_STATES, len(self.actions)])
        # set the conditions for converge in each case.
        if str == "Q-learning":
            threshold = 0.0001
            samples = 100
        elif str == "SARSA":
            threshold = 2
            samples = 500
        for i in range(1, NUM_OF_EPISODES):
            done = False
            self.state = START_STATE
            while not done:
              action = self.takeAnAction()
              action = self.checkIfActionIsValid(action)
              q_index=self.giveQindex()
              old_q_value = self.q_table[q_index][action]
              self.state = self.next_Position_mapped(action)
              self.next_Position_mapped(action)
              next_q_index=self.giveQindex()
              reward = self.reward_table[q_index][action]
              if str == "Q-learning":
                # Q-learning update
                next_q_value = np.max(self.q_table[next_q_index])
              elif str == "SARSA":
                # SARSA update
                next_action=self.takeAnAction()
                next_action = self.checkIfActionIsValid(next_action)
                next_q_value = self.q_table[next_q_index][next_action]
              new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_value) # Q-value update
              self.q_table[q_index][action] = new_q_value
              done = self.isDoneFunc()   #check if state is final
            self.cum_reward +=  reward
            if i%samples==0:
                if self.chechConverge(q_table_updated,threshold):
                    break
                q_table_updated = copy.deepcopy(self.q_table)
        print("Training finished.\n")
        print("Total episodes:", i, "\n")
        print("Cumulative reward:",self.cum_reward,"\n")
       # print("Performance:", self.cum_reward/NUM_OF_EPISODES,"\n")
        np.set_printoptions(formatter={'float': lambda x: " {0:0.3f} ".format(x)})
        print(self.actions)
        print(self.q_table)

    def takeAnAction(self):
         if random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.actions)  # Explore action space using greedypolicy
            action = self.mapActions(action)
         else:
            action = np.argmax(self.q_table[ self.state]) # Exploit learned values, take the best
         return action

    def giveQindex(self):
      # convert board index into Q-table-index
        q_table_index=self.state[0]+self.state[1]+3*self.state[0]
        return q_table_index

    def checkIfActionIsValid(self,action):
        # Check if the action is valid otherwise return a valid one
         temp_state = self.next_Position_mapped(action)
         while temp_state == self.state:  # check if taking the action the next state is valid
             action = self.takeAnAction() # if not try another action for current state
             temp_state = self.next_Position_mapped(action)
         valid_action=action
         return valid_action

    def mapActions(self,action):
        # Mapping Actions with index to be compatible with Q-table-column-index
        if action == "up":
            return 0
        elif action == "down":
            return 1
        elif action == "left":
            return 2
        else:
            return 3

    def next_Position_mapped(self, action):
        # next state in the board
        if action == 0:
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)
        # check if next state is valid
        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):
                if nxtState != (1, 1):
                    return nxtState
        # otherwise return the current state
        return self.state

    def isDoneFunc(self):
        #Done when end of state
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            return True

    def chechConverge(self,q_table_updated,threshold):
        converge = False
        difference = np.sum(abs(self.q_table - q_table_updated))
        if difference < threshold:
            converge = True
        return converge
