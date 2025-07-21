#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
import os

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        
        self.mean_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)
        
        init_layer_uniform(self.hidden1)
        init_layer_uniform(self.hidden2)
        init_layer_uniform(self.mean_layer)
        init_layer_uniform(self.log_std_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
    
        mean = self.mean_layer(x)
    
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)
        
        init_layer_uniform(self.hidden1)
        init_layer_uniform(self.hidden2)
        init_layer_uniform(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.value_layer(x)
        #############################

        return value

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
        
    def normalize(self, x):
        """Normalize the input using stored statistics."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]       # 17
        self.action_dim = env.action_space.shape[0]         # 6
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = args.test


        self.save_freq = args.save_freq
        self.checkpoint_path = args.checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.eval_interval = args.eval_interval
        self.eval_best_score = -float('inf')

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)

        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        # Clip actions to environment bounds
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        action = np.clip(action, action_low, action_high)

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done  # Return original reward for logging


    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)      # [2000, 17]
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            # Compute surrogate objectives for PPO-Clip
            surrogate1 = ratio * adv
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            # Add entropy bonus to encourage exploration
            entropy = dist.entropy().mean()
            actor_loss = actor_loss - self.entropy_weight * entropy
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            # Mean squared error between predicted values and returns
            value_pred = self.critic(state)
            critic_loss = F.mse_loss(value_pred, return_)
            #############################
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            wandb.log({
                "env steps": self.total_step,
                "Entropy": entropy,
            })

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        # self.is_test = False

        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)       #[1, 17]
        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        best_score = -float('inf')
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)
                
                state = next_state
                score += reward[0][0]
                # Inside episode end block:
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    wandb.log({
                        "episode": episode_count,
                        "Env step": self.total_step,
                        "reward": score,
                    })
                    # Save checkpoint periodically
                    if ep % self.save_freq == 0:
                        self.save_checkpoint(episode_count, score)
                    # Save best model
                    if score > best_score:
                        best_score = score
                        self.save_checkpoint(episode_count, score, best=True)
                    # Periodic evaluation
                    if self.total_step % self.eval_interval == 0:
                        eval_score = self.evaluate(20)
                        print(f"Evaluation after {ep} episodes: {eval_score:.2f}")
                        wandb.log({
                            "eval_score": eval_score,
                            "env_steps": self.total_step,
                        })
                        # Save best model
                        if eval_score > self.eval_best_score:
                            self.eval_best_score = eval_score
                            checkpoint = {
                                'actor_state_dict': self.actor.state_dict(),
                                'critic_state_dict': self.critic.state_dict(),
                                'actor_optimizer': self.actor_optimizer.state_dict(),
                                'critic_optimizer': self.critic_optimizer.state_dict(),
                            }
                            model_path = os.path.join(self.checkpoint_path, f"eval_best_model.pt")
                            torch.save(checkpoint, model_path)
                            print(f"New best model with score: {eval_score:.2f}")

                    score = 0
                    
            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            wandb.log({
                "env steps": self.total_step,
                "Critic loss": critic_loss,
                "Actor loss": actor_loss,
            })
            

        # termination
        self.env.close()
    def evaluate(self, num_episodes=5):
        """Evaluate the agent's performance."""
        # Save current state
        temp_is_test = self.is_test

        # Switch to test mode
        self.is_test = True
        total_reward = 0

        with torch.no_grad():
            for i in range(num_episodes):
                state, _ = self.env.reset(seed=self.seed+i)
                state = np.expand_dims(state, axis=0)  # Reshape state to match training
                episode_reward = 0
                done = False

                while not done:
                    action = self.select_action(state)
                    action = action.reshape(self.action_dim,)  # Fix action dimensions
                    next_state, reward, done = self.step(action)
                    state = next_state
                    episode_reward += reward[0][0]
                    done = done[0][0]

                total_reward += episode_reward
                print(f'seed {self.seed+i} got reward {episode_reward:.2f}')

        # Restore previous state
        self.is_test = temp_is_test

        return total_reward / num_episodes
    
    def test(self, video_folder: str):
        """Test the agent."""
        # self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env


    def save_checkpoint(self, episode, score, best=False):
        """Save model checkpoint.

        Args:
            episode (int): Current episode number
            score (float): Current episode score
            best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
            'score': score,
        }

        if best:
            model_path = os.path.join(self.checkpoint_path, f"best_model.pt")
            torch.save(checkpoint, model_path)
            print(f"Saved BEST checkpoint at episode {episode} with score {score:.2f}")
        else:
            model_path = os.path.join(self.checkpoint_path, f"checkpoint_ep{episode}.pt")
            torch.save(checkpoint, model_path)
            print(f"Saved checkpoint at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        """Load a saved model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run12")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=2000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-3) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=float, default=10)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--checkpoint-path", type=str, default="./result/ppo_walker12")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    seed = args.seed
    # random.seed(seed)
    # np.random.seed(seed)
    # seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    if not args.test:
        agent.train()
    else:
        agent.load_checkpoint(args.model_path)
        agent.evaluate(20)
        #agent.test("./result/ppo_walker_demo")