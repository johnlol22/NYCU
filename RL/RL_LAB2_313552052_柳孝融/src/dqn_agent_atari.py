import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
from models.dueling_model import dueling_model
import gymnasium as gym
from gymnasium.wrappers import FrameStack
import random
###
import cv2

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make("ALE/MsPacman-v5", render_mode = "rgb_array")
		self.env = gym.wrappers.AtariPreprocessing(self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.env = gym.wrappers.FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		self.test_env = gym.make("ALE/MsPacman-v5", render_mode = "rgb_array")
		self.test_env = gym.wrappers.AtariPreprocessing(self.test_env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		#self.load_and_evaluate('./log/DQN/Enduro/model_5215674_4448.pth')
		#self.load_and_evaluate('./log/DDQN/Enduro/model_16684165_4888.pth')
		
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		if random.random() < epsilon:
			action = np.random.randint(action_space.n)
		else:
			observation = np.array(observation)
			input = torch.FloatTensor(observation).unsqueeze(0)
			input = input.to(self.device)
			with torch.no_grad():
				action = torch.argmax(self.behavior_net(input)).item()
		return action

		#return NotImplementedError
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net
		
		#print(type(action)) #torch.tensor
		action = action.long()
		q_value = self.behavior_net(state).gather(1, action) #for dqn ddqn
		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].unsqueeze(1) #remind: for dqn
			'''
			best_action = self.behavior_net(next_state).argmax(1).unsqueeze(1) #remind: for ddqn
			q_next = self.target_net(next_state).gather(1, best_action)			#remind: for ddqn
			'''
			#if episode terminates at next_state, then q_target = reward
		q_target = reward + (1 - done) * self.gamma * q_next
        
		criterion = torch.nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	
	