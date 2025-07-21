#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple, List
import os

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)



class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        
        
        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.mu)
        #############################

        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        
        mu = torch.tanh(self.mu(x)) * 2.0
        
        std = torch.exp(self.log_std.clamp(-5, 2))
        
        dist = Normal(mu, std)
        
        action = dist.rsample()
        action = torch.clamp(action, -2.0, 2.0)
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)
        
        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.value)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value = self.value(x)
        #############################

        return value

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        # Update variance
        self.var = (self.count * self.var + batch_count * batch_var +
                   (delta**2) * self.count * batch_count / total_count) / total_count
        
        self.count = total_count
        
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
    

class A2CAgent:
    """A2CAgent interacting with environment with mini-batch updates."""

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.obs_rms = RunningMeanStd(env.observation_space.shape[0])

        # Mini-batch parameters
        self.batch_size = args.batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.n_steps = 4
        self.buffer_size = 0
        
        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = args.test

        self.save_freq = args.save_freq
        self.checkpoint_path = args.checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.reward_scale = 0.1  
        self.eval_interval = args.eval_interval

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        if not self.is_test:
            self.obs_rms.update(np.array([state]))
        state_norm = self.obs_rms.normalize(state)
        state_norm = torch.FloatTensor(state_norm).to(self.device)
        action, dist = self.actor(state_norm)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state_norm, log_prob]
            self.states.append(state)   # store the original state

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.log_probs.append(self.transition[1].detach().cpu().numpy())
            self.buffer_size += 1

        return next_state, reward, done

    def update_model_batch(self) -> Tuple[float, float]:
        """Update model with a batch of experiences."""
        # Process the batch
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards) / 10.0
        next_states = np.array(self.next_states)
        dones = np.array(self.dones)
        
        states_norm = self.obs_rms.normalize(states)
        next_states_norm = self.obs_rms.normalize(next_states)

        states_norm = torch.FloatTensor(states_norm).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states_norm = torch.FloatTensor(next_states_norm).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # Compute mask (1 - done)
        masks = 1 - dones
        
        # Get values
        values = self.critic(states_norm)
        with torch.no_grad():
            next_values = self.critic(next_states_norm)
        # Ensure correct dimensions
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)
        if len(dones.shape) == 1:
            masks = (1 - dones).unsqueeze(1)
        else:
            masks = 1 - dones

        # Compute target values
        target_values = rewards + self.gamma * next_values * masks
        
        # Compute value loss
        value_loss = F.mse_loss(values, target_values.detach())

        # Compute advantages
        advantages = (target_values.detach() - values.detach())

        # Normalize advantages for stability
        if advantages.shape[0] > 1:  # Only if batch size > 1
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        _, dist = self.actor(states_norm)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Make sure log_probs has correct shape
        if len(log_probs.shape) == 1:
            log_probs = log_probs.unsqueeze(1)
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_weight * entropy
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.buffer_size = 0

        
        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent using mini-batch updates."""
        step_count = 0
        best_score = -float('inf')
        for ep in tqdm(range(1, self.num_episodes)):
            # Clear memory at the start of each episode
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.log_probs.clear()
            self.buffer_size = 0
            
            # Reset environment
            state, _ = self.env.reset()
            score = 0
            done = False

            # Special for pendulum: set initial pendulum position upright with small noise
            # This makes learning much faster (curriculum learning)
            if ep < 100:  # First 100 episodes start near upright
                self.env.unwrapped.state = np.array([np.pi + np.random.normal(0, 0.1), 0])
            
            # Collect experience for this episode
            while not done:
                self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                
                state = next_state
                score += reward
                step_count += 1
                
                # Perform mini-batch updates when enough transitions are collected
                if self.buffer_size >= self.n_steps:
                    # Update model with batch
                    actor_loss, critic_loss = self.update_model_batch()
                    
                    # Log metrics
                    wandb.log({
                        "Env step": step_count,
                        "actor loss": actor_loss,
                        "critic loss": critic_loss,
                    })
            
            # Episode ended
            print(f"Episode {ep}: Total Reward = {score}")
            
            # Log episode results
            wandb.log({
                "episode": ep,
                "reward": score
            })
            
            # Save checkpoints
            if ep % self.save_freq == 0:
                self.save_checkpoint(ep, score)
            if ep % self.eval_interval == 0:
                eval_reward = self.evaluate(20)
                # Log episode results
                wandb.log({
                    "env steps": step_count,
                    "eval reward": eval_reward
                })
                # Save best model
                if eval_reward > best_score:
                    best_score = eval_reward
                    self.save_checkpoint(ep, score, best=True)


    def evaluate(self, num_episodes=20):
        """Evaluate the agent's performance."""
        # Save current state
        temp_is_test = self.is_test

        # Switch to test mode
        self.is_test = True
        total_reward = 0

        with torch.no_grad():
            for i in range(num_episodes):
                state, _ = self.env.reset(seed=self.seed+i)
                episode_reward = 0
                done = False

                while not done:
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)
                    state = next_state
                    episode_reward += reward
                    done = done

                print(f'seed {self.seed+i} has the reward {episode_reward:.2f}')
                total_reward += episode_reward

        # Restore previous state
        self.is_test = temp_is_test

        return total_reward / num_episodes
    

    def test(self, video_folder: str):
        """Test the agent."""
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        seed_list = [77, 78, 80, 81, 83, 84, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 103, 105]
        total_score = 0
        for i in seed_list:
            random.seed(i)
            np.random.seed(i)
            seed_torch(i)
            state, _ = self.env.reset(seed=i)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                
            print("score: ", score)
            total_score+=score
            self.env.close()

        self.env = tmp_env
        print(f'average reward: {total_score/len(seed_list)}')
    
    def save_checkpoint(self, episode, score, best=False):
        """Save model checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
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
        """Load a saved model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.obs_rms.mean = checkpoint['obs_rms_mean']
        self.obs_rms.var = checkpoint['obs_rms_var']
        return

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A2C for Pendulum-v1")
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-eval")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--checkpoint-path", type=str, default="./result/a2c_eval")
    parser.add_argument("--model-path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # Set up random seeds
    # setup_random_seeds(args.seed)
    
    # Initialize environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    # Initialize wandb
    wandb.init(
        project="DLP-Lab7-A2C-Pendulum", 
        name=args.wandb_run_name,
        config=vars(args),
        save_code=True
    )
    
    # Create and run agent
    agent = A2CAgent(env, args)
    
    if args.test and args.model_path:
        agent.load_checkpoint(args.model_path)
        agent.test("a2c_pendulum_test_videos")
    else:
        agent.train()