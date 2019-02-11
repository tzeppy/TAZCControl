from collections import deque
import numpy as np
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

log = logging.getLogger(__name__)


class ReplayMemory(object):
    def __init__(self, size):
        self.size = size
        self.storage = list()
        self.ptr = 0

    def push(self, item):
        if len(self.storage) < self.size:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
            self.ptr += 1
            self.ptr = self.ptr % self.size
        return self

    def sample(self, num_items):
        batch = random.sample(self.storage, num_items)
        return batch

    def __len__(self):
        return len(self.storage)


class DQN(nn.Module):
    def __init__(self, num_states, num_actions, h1=40, h2=15):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_states, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class BananaAgent(object):
    def __init__(self, model_name, state_size, action_size, memory=10000):
        self.model_name = model_name
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1
        # memory holds tuples (state, action, reward, next_state)
        self.replay_memory = ReplayMemory(memory)
        self.rewards = list()
        self.losses = deque(maxlen=100)
        self.dqn_local = DQN(state_size, action_size)
        self.dqn_target = DQN(state_size, action_size)
        self.dqn_copy()
        self.optimizer = Adam(self.dqn_local.parameters())
        self.at_step = 1
        self.learn_step = 1
        self.at_ep = 0
        self.learn_mod = 4
        self.batch_size = 32
        self.gamma = 0.99

    def reset_episode(self):
        self.at_ep += 1
        self.rewards = list()

    def sense(self, state, action, reward, next_state, learn=False):
        """observe result of action, potentially update brain"""
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        self.replay_memory.push((state, action, reward, next_state))
        self.rewards.append(reward)
        self.at_step += 1
        if learn is False:
            return
        if len(self.replay_memory) > self.batch_size and self.at_step % self.learn_mod == 0:
            # print(".", end="")  # noqa
            self.learn_a_batch()
            self.learn_step += 1

    def ave_loss(self):
        return sum(self.losses) / max(1, len(self.losses))

    def cum_rewards(self):
        return sum(self.rewards)

    def act(self, state, use_egreedy=False):
        """ e-greedy action """
        self.epsilon
        if use_egreedy and random.random() < self.epsilon:
            a = random.choice(range(self.action_size))
        else:
            state = torch.from_numpy(state).float()
            action_vals = self.dqn_local.forward(state)
            a = np.argmax(action_vals.data.numpy())
        return a

    def learn_a_batch(self):
        items = self.replay_memory.sample(self.batch_size)
        s0, a0, r1, s1 = zip(*items)
        s0 = torch.from_numpy(np.vstack([s for s in s0])).float()
        a0 = torch.from_numpy(np.vstack([a for a in a0])).long()
        r1 = torch.from_numpy(np.vstack([r for r in r1])).float()
        s1 = torch.from_numpy(np.vstack([s for s in s1])).float()

        q1 = self.dqn_target.forward(s1)
        q1_targets = r1 + (self.gamma * q1.max(1)[0].unsqueeze(1))  # max returns 2 things!

        q0 = self.dqn_local.forward(s0)
        q0_expected = q0.gather(1, a0)

        loss = F.mse_loss(q0_expected, q1_targets)
        self.optimizer.zero_grad()
        self.losses.append(loss)
        loss.backward()
        self.optimizer.step()
        if (self.learn_step % self.learn_mod) == 0:
            self.dqn_copy()

    def dqn_copy(self):
        # update target from local
        for tp, lp in zip(self.dqn_target.parameters(), self.dqn_local.parameters()):
            tp.data.copy_(lp.data)

    def load(self):
        fn = "{}.mdl".format(self.model_name)
        state_dict = torch.load(fn)
        self.dqn_local.load_state_dict(state_dict)
        self.dqn_target.load_state_dict(state_dict)
        log.info("loaded {}".format(fn))
        return self

    def save(self):
        fn = "{}.mdl".format(self.model_name)
        torch.save(self.dqn_local.state_dict(), fn)
        log.info("saved {}".format(fn))
        return self

    def __str__(self):
        return self.dqn_local
