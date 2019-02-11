
import logging
import numpy as np

log = logging.getLogger(__name__)

class RandomAgent(object):
	def __init__(self, model_name, state_size, action_size, memory=0):
		self.model_name = model_name
		self.state_size = state_size
		self.action_size = action_size
		self.rewards = list()

	def reset_episode(self):
		self.rewards = list()

	def sense(self, state, action, reward, next_state, learn=False):
		self.rewards.append(reward)

	def act(self, state, use_egreedy):
		a_vec = np.random.randn(1, self.action_size)
		a_vec = np.clip(a_vec, -1, 1)
		print(a_vec)
		return a_vec

	def load(self):
		pass

	def save(self):
		pass

	def cum_rewards(self):
		return sum(self.rewards)

	def ave_loss(self):
		return -1.0

	def __str__(self):
		return self.model_name
