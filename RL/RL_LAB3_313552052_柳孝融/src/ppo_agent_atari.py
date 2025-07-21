import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym

from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import torch.nn.functional as F

class AtariPPOAgent(PPOBaseAgent):
    def __init__(self, config):
        super(AtariPPOAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        # self.env = ???
        self.env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
        self.env = AtariPreprocessing(env=self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        self.env = FrameStack(env=self.env, num_stack=4, lz4_compress=False)
        ### TODO ###
        # initialize test_env
        # self.test_env = ???
        self.test_env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
        self.test_env = AtariPreprocessing(env=self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        self.test_env = FrameStack(env=self.env, num_stack=4, lz4_compress=False)

        self.net = AtariNet(self.env.action_space.n)
        self.net.to(self.device)
        #self.load_and_evaluate('./log6/Enduro_release/model_15791409_111.pth')
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
    def decide_agent_actions(self, observation, eval=False):
        ### TODO ###
        # add batch dimension in observation
        observation = np.array(observation)
        observation = torch.tensor(observation)
        observation = observation.to(self.device)
        # get action, value, logp from net
		
        # if eval:
        # 	with torch.no_grad():
        # 		???, ???, ???, _ = self.net(observation, eval=True)
        # else:
        # 	???, ???, ???, _ = self.net(observation)
        if eval:
            with torch.no_grad():
                action, logp, value, entropy = self.net(observation, eval=True)
        else:
            observation = torch.unsqueeze(observation, 0)
            action, logp, value, entropy = self.net(observation)
        return action.cpu().detach(), value.cpu().detach(), logp.cpu().detach()

	
    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
        sample_count = len(batches["action"])
        batch_index = np.random.permutation(sample_count)
		
        observation_batch = {}
        for key in batches["observation"]:  #key = pixel, pos, ...
            observation_batch[key] = batches["observation"][key][batch_index]  #observation keys like, pixel and pos, are stored randomly according to the index of batch_index
        action_batch = batches["action"][batch_index]                           #so as to action, return, value, log_pi
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]
        

        for _ in range(self.update_count): #update_count = 3
            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
                ac_train_batch = action_batch[start:start + self.batch_size]
                return_train_batch = return_batch[start:start + self.batch_size]
                adv_train_batch = adv_batch[start:start + self.batch_size]
                v_train_batch = v_batch[start:start + self.batch_size]
                logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

                ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
                ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
                ac_train_batch = torch.from_numpy(ac_train_batch)
                ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)
                ### TODO ###
                # calculate loss and update network
                _, action_probability, value, entropy = self.net(ob_train_batch, False, torch.squeeze(ac_train_batch))
                # calculate policy loss
                # ratio = ???
                # surrogate_loss = ???
                logp_pi_train_batch = torch.squeeze(logp_pi_train_batch)
                ratio = action_probability - logp_pi_train_batch #action_probability is new policy?
                ratio = torch.exp(ratio)

                surrogate_loss1 = ratio*adv_train_batch
                surrogate_loss2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_train_batch
                surrogate_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

                entropy = torch.mean(entropy, dtype=torch.float32)
                # calculate value loss
                # value_criterion = nn.MSELoss()
                # v_loss = value_criterion(...)

                value_criterion = nn.MSELoss()
                v_loss = value_criterion(value, return_train_batch)
                # calculate total loss
                # loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
                loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
                # update network
                # self.optim.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
                # self.optim.step()
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
                self.optim.step()

                # total_surrogate_loss += surrogate_loss.item()
                # total_v_loss += v_loss.item()
                # total_entropy += entropy.item()
                # total_loss += loss.item()
                # loss_counter += 1
                total_surrogate_loss += surrogate_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                loss_counter += 1

        self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
        self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
        self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
        self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
        print(f"Loss: {total_loss / loss_counter}\
            \tSurrogate Loss: {total_surrogate_loss / loss_counter}\
            \tValue Loss: {total_v_loss / loss_counter}\
            \tEntropy: {total_entropy / loss_counter}\
            ")
