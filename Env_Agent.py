import random
import math
import gym
import numpy as np
import matplotlib.pyplot as plt

class Agent():
 def __init__(self, env, algorithm, alpha, gamma, epsilon, num_episodes ):
   self.env = env
   self.algorithm = algorithm
   self.alpha = alpha
   self.gamma = gamma
   self.epsilon = epsilon
   self.num_episodes = num_episodes

 def discretize(self,observation):
     if self.env.spec.id == 'CartPole-v0' or env.spec.id == 'CartPole-v1':
         self.buckets = (1, 1, 6, 12,)
         upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
         lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
     elif self.env.spec.id == 'MountainCar-v0':
         self.buckets = (16, 16,)
         upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
         lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
     ratios = [(observation[i] + abs(lower_bounds[i]))/(upper_bounds[i] -lower_bounds[i]) for i in range(len(observation))]
     new_observation = [int(round(self.buckets[i]-1)*ratios[i]) for i in range(len(observation))]
     new_observation = [min(self.buckets[i] - 1, max(0, new_observation[i])) for i in range(len(observation))]
     return tuple(new_observation)

 def takeGreedyAction(self,state):
     if random.uniform(0, 1) < self.epsilon:
         action =self.env.action_space.sample()  # Explore action space using greedypolicy
     else:
         action = np.argmax(self.q_table[state])  # Exploit learned values, take the best
     return action

 def QtableUpdate(self,current_state,action,new_state,reward):
    if self.algorithm == 'Q-learning':
        # Q-learning update
        new_q_value = np.max(self.q_table[new_state])
    elif self.algorithm == 'SARSA':
        # SARSA update
        new_action = self.takeGreedyAction(new_state)
        new_q_value = self.q_table[new_state][new_action]
    elif self.algorithm == 'Expected SARSA':
        # Expected SARSA update
        new_q_value = np.mean(self.q_table[new_state][:])
    old_q_value = self.q_table[current_state][action]
    updated_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * new_q_value)
    self.q_table[current_state][action] = updated_q_value

 def simulate(self):
     episodes_array=np.arange(0,self.num_episodes)
     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
     plt.subplots_adjust(wspace=0.5)
     axes[0].plot(self.episodes_rewards)
     axes[0].set_title('Reward per episode')
     axes[0].set_ylabel('Reward')
     axes[1].plot(self.episodes_lengths)
     axes[1].set_title('Length per episode')
     axes[1].set_ylabel('Length')
     plt.show()

 def train(self):
     current_state = self.discretize(self.env.reset())
     self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))
     total_reward = 0
     self.episodes_lengths = np.zeros(self.num_episodes)
     self.episodes_rewards = np.zeros(self.num_episodes)
     for i in range(0, self.num_episodes):
         done = False
         episode_reward = 0
         current_state = self.discretize(self.env.reset())
         counter = 0
         while not done:
             action = self.takeGreedyAction(current_state)
             obs, reward, done, info = self.env.step(action)
             episode_reward +=reward
             new_state = self.discretize(obs)
             self.QtableUpdate(current_state,action,new_state,reward)
             current_state = new_state
             counter += 1
         total_reward += episode_reward
         if episode_reward>-200 : print(f"The episode {i} took reward: {episode_reward}")
         self.episodes_rewards[i] = episode_reward
         self.episodes_lengths[i] = counter

 def play(self, rounds=100, N=3):
     # N-steps
     debug = False
     for _ in range(rounds):
         current_state = self.discretize(self.env.reset())   #self.reset()
         t = 0
         T = 0
         action = self.takeGreedyAction(current_state) #action = self.chooseAction()
         actions = [action]
         states = [current_state]
         rewards = [0]
         while True:
             if t < T:
                 state, reward, done, info = self.env.step(action)
                 state = self.discretize(state)
                 states.append(state)
                 rewards.append(reward)
                 if done:
                         print("End at state {} | number of states {}".format(state, len(states)))
                         T = t + 1
                 else:
                     action = self.takeGreedyAction(state)
                     actions.append(action)
             # state tau being updated
             tau = t - N + 1
             if tau >= 0:
                 G = 0
                 for i in range(tau + 1, min(tau + N + 1, T + 1)):
                     G += np.power(gamma, i - tau - 1) * rewards[i]
                 if tau + N < T:
                     state_action = (states[tau + N], actions[tau + N])
                     G += np.power(gamma, N) * self.q_table[state_action[0]][state_action[1]]
                 state_action = (states[tau], actions[tau])
                 self.q_table[state_action[0]][state_action[1]] += alpha * (G - self.q_table[state_action[0]][state_action[1]])
             if tau == T - 1:
                 break
             t += 1
         print(rewards)

env = gym.make('MountainCar-v0')
agent = Agent(env,algorithm='Q-learning',alpha = 0.08,gamma = 0.8,epsilon = 0.35,num_episodes = 2500) #Env options: 'MountainCar-v0' 'CartPole-v0' Algorithms options: "Q-learning" "SARSA" "Expected SARSA"
agent.train()
agent.simulate()
#agent.play()
env.close()
