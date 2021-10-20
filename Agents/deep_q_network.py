import gym
import numpy as np
import torch
from Agents.generic_agents import Agent


class Deep_Q_Agent(Agent):
    def __init__(self, envs, name, type, num_episodes, gamma, lr=0.003, eps=0.1, anneal_lr_param=1.,
                 anneal_epsilon_param=1., threshold_lr_anneal=100., evaluate_every_n_episodes=20):
        super(Deep_Q_Agent, self).__init__(envs, name, type, num_episodes, gamma, lr, anneal_lr_param,
                                           evaluate_every_n_episodes)
        hidden_size = 256
        obs_size = self.train_env.action_space.nvec[0]
        n_actions = self.train_env.observation_space.low.size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions),
            torch.nn.Softmax(dim=0)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.horizon = 500
        self.max_trajectories = 500
        self.gamma = gamma
        self.score = []

    def train(self):
        for trajectory in range(self.max_trajectories):
            state = self.train_env.reset()
            done = False
            transitions = []
            for t in range(self.horizon):
                action = self.choose_action(state, env=self.train_env)
                prev_state = state
                state, _, done, info = self.train_env.step(action)
                transitions.append((prev_state, action, t + 1))
                if done:
                    break
            self.score.append(len(transitions))
            reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))

            batch_gvals = []
            for i in range(len(transitions)):
                new_gval = 0
                power = 0
                for j in range(i, len(transitions)):
                    new_gval = new_gval + ((self.gamma ** power) * reward_batch[j]).numpy()
                    power += 1
                batch_gvals.append(new_gval)
            expected_returns_batch = torch.FloatTensor(batch_gvals)

            expected_returns_batch /= expected_returns_batch.max()

            state_batch = torch.Tensor([s for (s, a, r) in transitions])
            action_batch = torch.Tensor([a for (s, a, r) in transitions])

            pred_batch = self.model(state_batch)
            prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

            loss = - torch.sum(torch.log(prob_batch) * expected_returns_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if trajectory % 50 == 0 and trajectory > 0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(self.score[-50:-1])))

    def choose_action(self, state, env):
        act_prob = self.model(torch.from_numpy(state).float())
        action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
        return action
