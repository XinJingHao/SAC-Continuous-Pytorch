import numpy as np
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomBuffer(object):
	def __init__(self, state_dim, action_dim, Env_with_dead , max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.Env_with_dead = Env_with_dead

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dead = np.zeros((max_size, 1),dtype=np.uint8)

		self.device = device


	def add(self, state, action, reward, next_state, dead):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		# it is important to distinguish between dead and done!!!
		# See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
		if self.Env_with_dead:
			self.dead[self.ptr] = dead
		else:
			self.dead[self.ptr] = False

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		with torch.no_grad():
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.dead[ind]).to(self.device)
			)

	def save(self):
		'''save the replay buffer if you want'''
		scaller = np.array([self.max_size,self.ptr,self.size,self.Env_with_dead],dtype=np.uint32)
		np.save("buffer/scaller.npy",scaller)
		np.save("buffer/state.npy", self.state)
		np.save("buffer/action.npy", self.action)
		np.save("buffer/reward.npy", self.reward)
		np.save("buffer/next_state.npy", self.next_state)
		np.save("buffer/dead.npy", self.dead)

	def load(self):
		scaller = np.load("buffer/scaller.npy")

		self.max_size = scaller[0]
		self.ptr = scaller[1]
		self.size = scaller[2]
		self.Env_with_dead = scaller[3]

		self.state = np.load("buffer/state.npy")
		self.action = np.load("buffer/action.npy")
		self.reward = np.load("buffer/reward.npy")
		self.next_state = np.load("buffer/next_state.npy")
		self.dead = np.load("buffer/dead.npy")

