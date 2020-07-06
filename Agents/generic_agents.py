from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np


class Agent():

    def __init__(self, env, num_episodes, gamma):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.gamma_initial = gamma

    def train(self):
        self.total_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            self.done = False
            episode_reward = 0
            current_state = self.env.reset()
            self.counter = 0
            while not self.done:
                action = self.choose_action(current_state)
                new_state, reward, self.done, info = self.env.step(action)
                episode_reward += reward
                self.done=self.update(current_state, action, new_state, reward, self.done, i,self.counter)
                current_state = deepcopy(new_state)
                self.counter += 1
            self.total_rewards[i] = episode_reward
            if i % 500 == 0: self.evaluate()

    def choose_action(self, state):
        return 1

    def choose_best_action(self, state):
        return 1

    def evaluate(self):
        done = False
        episode_reward = 0
        current_state = self.env.reset()
        counter = 0
        while not done:
            action = self.choose_best_action(current_state)
            # self.env.render()
            new_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            current_state = deepcopy(new_state)
            counter += 1
        print("Reward on evaluation %f.4" % episode_reward)

    def update(self, state, action, new_state, reward, done, current_episode=1, episode_length=1):
        pass

    def plot(self, exp_path):
        mean_rewards = np.zeros(self.num_episodes)
        for i in range(self.num_episodes):
            mean_rewards[i] = np.mean(self.total_rewards[max(0, i - 50):(i + 1)])
        plt.plot(mean_rewards)
        plt.savefig('%s/%s on %s for %d episodes with %d epsilon %d gamma %d alpha and %s stepsizes.png' % (exp_path, self.__class__.__name__, self.env.spec.id, self.num_episodes,self.epsilon_initial,self.gamma_initial,self.alpha_initial,self.env.observation_space.nvec))
