
from collections import deque
import logging
import numpy as np
import random
import time

from model import Actor, Critic
from utils import ReplayBuffer, OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim


log = logging.getLogger(__name__)

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, model_name, state_size, action_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.model_name = model_name
        self.state_size = state_size
        self.action_size = action_size
        random_seed = int(time.time())
        self.seed = random.seed(random_seed)
        self.rewards = list()
        self.losses = deque(maxlen=100)
        self.stepn = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def sense(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.stepn += 1
        if reward or self.stepn % 2 == 0:
            self.memory.add(state, action, reward, next_state, done)
        self.rewards.append(reward)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            if self.stepn % 5 == 0:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_episode(self):
        self.rewards = list()
        self.noise.reset()
        self.stepn = 0

    def ave_loss(self):
        return sum(self.losses) / max(1, len(self.losses))

    def cum_rewards(self):
        return sum(self.rewards)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.losses.append(actor_loss)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load(self):
        afn = "{}_actor.mdl".format(self.model_name)
        cfn = "{}_critic.mdl".format(self.model_name)
        state_dict = torch.load(afn)
        self.actor_local.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)
        state_dict = torch.load(cfn)
        self.critic_local.load_state_dict(state_dict)
        self.critic_target.load_state_dict(state_dict)
        log.info("loaded {}, {}".format(afn, cfn))
        return self

    def save(self):
        afn = "{}_actor.mdl".format(self.model_name)
        cfn = "{}_critic.mdl".format(self.model_name)
        torch.save(self.actor_local.state_dict(), afn)
        torch.save(self.critic_local.state_dict(), cfn)
        log.info("saved to {}, {}".format(afn, cfn))
        return self
