import random
import math
import gym
import numpy as np

NUM_OF_EPISODES = 1000
alpha = 0.7
gamma = 0.85
epsilon = 0.25

class Env_Agent():
 def __init__(self, env):
   self.env = env
   if env.spec.id == 'CartPole-v0' or env.spec.id == 'CartPole-v1':
       self.buckets = (1, 1, 6, 12,)
   elif env.spec.id == 'MountainCar-v0':
       self.buckets = (1, 1,)
   self.q_table = np.zeros(self.buckets + (env.action_space.n,))

 def discretize(self,observation):
     if self.env.spec.id == 'CartPole-v0' or env.spec.id == 'CartPole-v1':
         upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
         lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
     elif self.env.spec.id == 'MountainCar-v0':
         upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
         lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
     ratios = [(observation[i] + abs(lower_bounds[i]))/(upper_bounds[i] -lower_bounds[i]) for i in range(len(observation))]
     new_observation = [int(round(self.buckets[i]-1)*ratios[i]) for i in range(len(observation))]
     new_observation = [min(self.buckets[i] - 1, max(0, new_observation[i])) for i in range(len(observation))]
     return tuple(new_observation)

 def takeAnAction(self,state):
     if random.uniform(0, 1) < epsilon:
         action =self.env.action_space.sample()  # Explore action space using greedypolicy
     else:
         action = np.argmax(self.q_table[state])  # Exploit learned values, take the best
     return action

 def Q_values(self, str):
     for i in range(1, NUM_OF_EPISODES):
         done = False
         total_reward = 0
         current_state = self.discretize(self.env.reset())
         while not done:
             self.env.render()
             action = self.takeAnAction(current_state)
             obs, reward, done, info = self.env.step(action)
             total_reward +=reward
             new_state = self.discretize(obs)
             old_q_value = self.q_table[current_state][action]
             if str == 'Q-learning':
                 # Q-learning update
                 new_q_value = np.max(self.q_table[new_state])
             elif str == 'SARSA':
                 # SARSA update
                 new_action = self.takeAnAction(new_state)
                 new_q_value = self.q_table[new_state][new_action]
             elif str == 'Expected SARSA':
                 # Expected SARSA update
                 new_q_value = np.mean(self.q_table[new_state][:])
             updated_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * new_q_value)
             self.q_table[current_state][action] = updated_q_value
             current_state = new_state
         if total_reward>-200 :(f"The episode {i} took reward: {total_reward}")
         #print(obs)


env = gym.make('CartPole-v0') #options: 'MountainCar-v0' 'CartPole-v0'
agent = Env_Agent(env)
agent.Q_values('Q-learning') #options: "Q-learning" "SARSA" "Expected SARSA"
env.close()