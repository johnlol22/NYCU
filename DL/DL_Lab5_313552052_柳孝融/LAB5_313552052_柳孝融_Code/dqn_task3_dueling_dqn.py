# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, num_actions, input_channels=4):
        super(DQN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals



class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)



class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # If buffer not full, add new item
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        # Otherwise, replace item at current position
        else:
            self.buffer[self.pos] = transition
        priority = self.max_priority if error is None else (np.abs(error) + 1e-5) ** self.alpha
        
        # Store priority at current position
        self.priorities[self.pos] = priority
        
        # Move position pointer and wrap around if necessary
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        N = len(self.buffer)
        
        if N == 0:
            return None, None, None
            
        # Get priorities for items in buffer (not empty slots)
        priorities = self.priorities[:N]
        
        # Convert priorities to probabilities through normalization
        probs = priorities / np.sum(priorities)
        
        # Sample indices based on these probabilities
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        
        # Retrieve samples from buffer
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        
        # Normalize weights to scale updates correctly
        weights = weights / np.max(weights)
        
        # Return samples, their indices (for later priority updates), and their weights
        return samples, indices, weights
        ########## END OF YOUR CODE (for Task 3) ########## 

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            # Ensure index is within bounds
            if idx < len(self.buffer):
                # Calculate priority from error (add small constant to avoid zero priority)
                priority = (np.abs(error) + 1e-5) ** self.alpha
                
                # Update priority in the priorities array
                self.priorities[idx] = priority
                
                # Update max priority if needed
                self.max_priority = max(self.max_priority, priority)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def __len__(self):
        return len(self.buffer)

# ablation study
# class MultiStepReplayBuffer:
#     def __init__(self, capacity, n_steps=3, gamma=0.99):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#         self.n_steps = n_steps
#         self.gamma = gamma
#         self.recent_transitions = deque(maxlen=n_steps)
#         
#     def add(self, state, action, reward, next_state, done):
#         # Add the new transition to the recent transitions buffer
#         self.recent_transitions.append((state, action, reward, next_state, done))
#         
#         # If we have enough transitions, compute n-step return and add to buffer
#         if len(self.recent_transitions) == self.n_steps:
#             n_step_return, final_state, final_done = self._compute_n_step_return()
#             
#             # Get the initial state and action from the oldest transition
#             initial_state, initial_action, _, _, _ = self.recent_transitions[0]
#             
#             # Store the n-step transition
#             transition = (initial_state, initial_action, n_step_return, final_state, final_done)
#             
#             if len(self.buffer) < self.capacity:
#                 self.buffer.append(transition)
#             else:
#                 self.buffer[self.position] = transition
#                 
#             self.position = (self.position + 1) % self.capacity
#         
#         # If episode ended, flush the recent transitions
#         if done:
#             while len(self.recent_transitions) > 0:
#                 n_step_return, final_state, final_done = self._compute_n_step_return()
#                 initial_state, initial_action, _, _, _ = self.recent_transitions[0]
#                 
#                 transition = (initial_state, initial_action, n_step_return, final_state, final_done)
#                 
#                 if len(self.buffer) < self.capacity:
#                     self.buffer.append(transition)
#                 else:
#                     self.buffer[self.position] = transition
#                     
#                 self.position = (self.position + 1) % self.capacity
#                 self.recent_transitions.popleft()
#     
#     def _compute_n_step_return(self):
#         # Compute n-step discounted return: R_t + gamma*R_{t+1} + ... + gamma^(n-1)*R_{t+n-1}
#         n_step_reward = 0
#         final_state = None
#         final_done = False
#         
#         for i, (_, _, r, next_s, d) in enumerate(self.recent_transitions):
#             n_step_reward += (self.gamma ** i) * r
#             if d:
#                 final_state = next_s
#                 final_done = True
#                 break
#             final_state = next_s
#             
#         return n_step_reward, final_state, final_done
#     
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return states, actions, rewards, next_states, dones
#     
#     def __len__(self):
#         return len(self.buffer)


# Combining with Prioritized Experience Replay
class MultiStepPrioritizedReplayBuffer:
    def __init__(self, capacity, n_steps=3, gamma=0.99, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Increment beta over time
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        self.recent_transitions = deque(maxlen=n_steps)
        
    def add(self, state, action, reward, next_state, done, error=None):
        # Add to recent transitions
        self.recent_transitions.append((state, action, reward, next_state, done))
        
        # Process if we have enough transitions or episode ended
        if len(self.recent_transitions) == self.n_steps or done:
            n_step_return, final_state, final_done = self._compute_n_step_return()
            initial_state, initial_action, _, _, _ = self.recent_transitions[0]
            
            # Create transition with n-step return
            transition = (initial_state, initial_action, n_step_return, final_state, final_done)
            
            # Add to buffer
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
                
            # Set priority
            priority = self.max_priority if error is None else (np.abs(error) + 1e-5) ** self.alpha
            self.priorities[self.pos] = priority
            
            self.pos = (self.pos + 1) % self.capacity
            
            # If episode ended, process remaining transitions
            if done:
                self.recent_transitions.popleft()
                while len(self.recent_transitions) > 0:
                    n_step_return, final_state, final_done = self._compute_n_step_return()
                    initial_state, initial_action, _, _, _ = self.recent_transitions[0]
                    
                    transition = (initial_state, initial_action, n_step_return, final_state, final_done)
                    
                    if len(self.buffer) < self.capacity:
                        self.buffer.append(transition)
                    else:
                        self.buffer[self.pos] = transition
                    
                    # Set priority    
                    self.priorities[self.pos] = self.max_priority if error is None else (np.abs(error) + 1e-5) ** self.alpha
                    self.pos = (self.pos + 1) % self.capacity
                    self.recent_transitions.popleft()
    
    def _compute_n_step_return(self):
        # Compute n-step discounted return
        n_step_reward = 0
        final_state = None
        final_done = False
        
        for i, (_, _, r, next_s, d) in enumerate(self.recent_transitions):
            n_step_reward += (self.gamma ** i) * r
            if d:
                final_state = next_s
                final_done = True
                break
            final_state = next_s
            
        return n_step_reward, final_state, final_done
    
    def sample(self, batch_size):
        # Update beta parameter
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        N = len(self.buffer)
        
        if N == 0:
            return None, None, None
            
        priorities = self.priorities[:N]
        probs = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        
        # Retrieve samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # Extract components
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.buffer):
                priority = (np.abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)


        self.criterion = nn.MSELoss()  # Using MSE loss for DQN
        # self.memory = MultiStepReplayBuffer(
        #     capacity=args.memory_size,
        #     n_steps=3,
        #     gamma=self.gamma
        # )
        # self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)
        self.memory = MultiStepPrioritizedReplayBuffer(
            capacity=args.memory_size,
            n_steps=3,  # Shorter n-step return for faster learning
            gamma=self.gamma,
            alpha=0.7,  # Higher alpha for more aggressive prioritization
            beta=0.4    # Start with lower beta
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=500):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)    # 4, 84, 84
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                # Frame skipping
                skip_frames = 4
                total_reward = 0
                last_obs = None

                for i in range(skip_frames):
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    last_obs = next_obs  # Store the last observation
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break

                next_state = self.preprocessor.step(last_obs)
                clipped_reward = np.clip(total_reward, -1, 1)
                # self.memory.add(state, action, clipped_reward, next_state, done)
                # self.memory.append((state, action, clipped_reward, next_state, done))
                # for all
                self.memory.add(state, action, clipped_reward, next_state, done, error=None)
                # for ddqn, prioritized
                # self.memory.add((state, action, clipped_reward, next_state, done), error=None)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    wandb.log({
                        "Memory Size": len(self.memory),
                        "Last Reward": reward,
                        "Current Episode Reward": total_reward
                    })
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            wandb.log({
                "Episode Length": step_count,
                "Memory Buffer Size": len(self.memory)
            })
            ########## END OF YOUR CODE ##########  
            if ep % 5 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 5 == 0:
                total_eval_reward = 0
                for i in range(5):
                    eval_reward = self.evaluate()
                    total_eval_reward+=eval_reward
                eval_reward = total_eval_reward / 5.
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
    
        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            
            # Frame skipping during evaluation
            skip_frames = 4
            skip_reward = 0
            last_obs = None
            
            for i in range(skip_frames):
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                last_obs = next_obs
                skip_reward += reward
                done = terminated or truncated
                if done:
                    break
                    
            total_reward += skip_reward
            state = self.preprocessor.step(last_obs)
    
        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        # In train method
        if self.epsilon > self.epsilon_min:
            # Decay epsilon faster in the beginning
            if self.train_count < 50000:
                self.epsilon *= 0.999
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)
        self.train_count += 1
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        # samples, indices, weights = self.memory.sample(self.batch_size)
        # states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.memory.n_steps) * next_q_values
            # target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        td_errors = np.power(td_errors + 1e-6, 1.0)  # Higher power for more prioritization
        # Compute weighted MSE loss to correct for PER bias
        loss = (weights * ((q_values - target_q_values) ** 2)).mean()
        # loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        self.memory.update_priorities(indices, td_errors)
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.detach().item():.4f} Q mean: {q_values.mean().detach().item():.3f} std: {q_values.std().detach().item():.3f}")
        wandb.log({
            "Loss": loss.detach().item(),
            "q value": q_values.mean().detach().item(),
            "std": q_values.std().detach().item()
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/task3_3")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run_task3_3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--train-per-step", type=int, default=4)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong_task3_3", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()

'''
/home/johnlol/nas/home/DL_Lab5/dqn_task2.py:305: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/aten/src/ATen/native/Scalar.cpp:22.)
  print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
'''