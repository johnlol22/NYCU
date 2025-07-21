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
        
        # Improved feature extractor with slightly more filters
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Increased from 64 to 128
            nn.ReLU()
        )
        
        # Value stream with wider architecture
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 128, 512),  # Adjusted for increased input
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream with wider architecture
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 128, 512),  # Adjusted for increased input
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Improved Dueling DQN combination
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

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition, error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        priority = self.max_priority if error is None else (np.abs(error) + 1e-5) ** self.alpha
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        return 
        
    def sample(self, batch_size):
        N = len(self.buffer)
        
        if N == 0:
            return None, None, None
        priorities = self.priorities[:N]
        
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.buffer):
                priority = (np.abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
        return
        
    def __len__(self):
        return len(self.buffer)


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
        # Create multiple environments for more diverse experiences
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Initialize networks with improved architecture
        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Higher Adam epsilon for better stability in gradient updates
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1e-3)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        
        # Track progress more carefully
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        self.avg_reward = -21  # Track average reward for more stable evaluation
        self.recent_rewards = deque(maxlen=10)  # Track recent rewards for adaptive training
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        
        # Create directories for model checkpoints at specific step intervals
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.milestone_checkpoints = {
            200000: False,
            400000: False, 
            600000: False,
            800000: False,
            1000000: False
        }

        # Use a combination of Huber and MSE loss for better stability
        self.mse_criterion = nn.MSELoss(reduction='none')
        self.huber_criterion = nn.SmoothL1Loss(reduction='none')
        
        # Use enhanced multi-step prioritized replay with tuned n_steps
        self.memory = MultiStepPrioritizedReplayBuffer(
            capacity=args.memory_size,
            n_steps=args.n_steps,
            gamma=self.gamma,
            alpha=args.per_alpha,
            beta=args.per_beta
        )
        
        # Use a more targeted LR scheduler based on looking at the plot patterns
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, 
        #     milestones=[50000, 150000, 300000], 
        #     gamma=0.5
        # )
        
        # Adaptive target update frequency - starts high, then reduces
        self.initial_target_update = args.target_update_frequency
        self.min_target_update = max(100, args.target_update_frequency // 10)
        self.current_target_update = self.initial_target_update

    def select_action(self, state):
        # Progressive exploration strategy
        exploration_phase = min(1.0, self.env_count / 500000)
        
        # Early phase: mix of epsilon-greedy and Boltzmann
        if exploration_phase < 0.3:
            # Use Boltzmann exploration occasionally for better state-dependent exploration
            if random.random() < 0.3:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_net(state_tensor)[0]
                    # Apply temperature to softmax
                    temperature = max(0.5, 1.0 - exploration_phase * 2)  # Starts high, decreases
                    probs = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
                    return np.random.choice(len(probs), p=probs)
            
            # Otherwise, use more aggressive epsilon-greedy
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
        
        # Mid phase: standard epsilon-greedy
        elif exploration_phase < 0.7:
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
        
        # Late phase: epsilon-greedy with action-value weighted randomness when exploring
        else:
            if random.random() < self.epsilon:
                # Most of the time use pure random
                if random.random() < 0.8:
                    return random.randint(0, self.num_actions - 1)
                # Sometimes use value-weighted exploration for better exploration
                else:
                    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.q_net(state_tensor)[0].cpu().numpy()
                        # Convert to probabilities with a minimum chance
                        q_min = q_values.min()
                        q_values = q_values - q_min + 1e-5
                        probs = q_values / q_values.sum()
                        return np.random.choice(len(probs), p=probs)
        
        # Exploitation: use Q-values to select best action
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1200):
        # Enhanced warmup phase - collect experiences with more diverse actions
        if self.env_count == 0:
            print("Starting enhanced warmup phase...")
            while len(self.memory) < self.replay_start_size:
                obs, _ = self.env.reset()
                state = self.preprocessor.reset(obs)
                done = False
                episode_steps = 0

                while not done and episode_steps < 2000:  # Limit episode length during warmup
                    # Occasionally take non-random actions even during warmup
                    # This helps create better initial experiences
                    if len(self.memory) > self.replay_start_size // 2 and random.random() < 0.2:
                        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            q_values = self.q_net(state_tensor)
                        action = q_values.argmax().item()
                    else:
                        # Take random action with slight bias toward promising actions
                        action = random.randint(0, self.num_actions - 1)
                    
                    # Frame skipping during warmup too for consistency
                    skip_frames = 2
                    skip_reward = 0
                    last_obs = None
                    
                    for i in range(skip_frames):
                        next_obs, reward, terminated, truncated, _ = self.env.step(action)
                        last_obs = next_obs
                        skip_reward += reward
                        done = terminated or truncated
                        if done:
                            break
                            
                    next_state = self.preprocessor.step(last_obs)
                    
                    # Apply reward shaping during warmup too
                    shaped_reward = self._shape_reward(skip_reward)
                    
                    # Better error estimate for initial experiences
                    if len(self.memory) > 1000 and len(self.memory) % 20 == 0:
                        # Occasionally compute TD errors for better initial priorities
                        with torch.no_grad():
                            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                            next_state_tensor = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(self.device)
                            
                            q_value = self.q_net(state_tensor)[0, action]
                            next_q = self.target_net(next_state_tensor).max()
                            target = shaped_reward + (1 - done) * (self.gamma ** self.memory.n_steps) * next_q
                            error = abs(q_value - target).item()
                    else:
                        error = None
                    
                    self.memory.add(state, action, shaped_reward, next_state, done, error=error)
                    state = next_state
                    self.env_count += 1
                    episode_steps += 1

                    # Occasionally train during warmup to improve initial performance
                    if len(self.memory) > self.batch_size * 2 and self.env_count % 10 == 0:
                        self.train()

                    if len(self.memory) >= self.replay_start_size:
                        break
                        
            print(f"Warmup complete! Collected {len(self.memory)} experiences")
            
            # Pre-train the network for a bit to bootstrap learning
            print("Pre-training network...")
            for _ in range(1000):
                self.train()
            print("Pre-training complete")
        
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            raw_total_reward = 0  # Keep track of unclipped rewards
            step_count = 0
            episode_start_time = time.time()
            
            # Episode-specific adaptations based on current performance
            # Adaptive train_per_step based on current performance
            if self.avg_reward < -15:  # If performing poorly, train less
                effective_train_per_step = min(8, self.train_per_step * 2)
            elif self.avg_reward > 5:  # If performing well, train more
                effective_train_per_step = max(1, self.train_per_step // 2)
            else:
                effective_train_per_step = self.train_per_step

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                
                # Frame skipping with variable frame skip
                # Use more frame skipping when we're confident in our policy
                if self.avg_reward > 0:
                    skip_frames = 4
                else:
                    skip_frames = 1  # Less skipping when learning
                    
                skip_reward = 0
                last_obs = None
                experiences = []  # Collect all frame experiences for potential use

                for i in range(skip_frames):
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    experiences.append((next_obs, reward, terminated, truncated))
                    last_obs = next_obs
                    skip_reward += reward
                    done = terminated or truncated
                    if done:
                        break

                next_state = self.preprocessor.step(last_obs)
                
                # Apply reward shaping
                shaped_reward = self._shape_reward(skip_reward)
                
                # Add to replay buffer with shaped reward
                # Compute error estimate for better prioritization
                if self.env_count % 5 == 0:  # Occasionally compute actual errors
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                        next_state_tensor = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(self.device)
                        
                        q_value = self.q_net(state_tensor)[0, action]
                        next_q = self.target_net(next_state_tensor).max()
                        target = shaped_reward + (1 - done) * (self.gamma ** self.memory.n_steps) * next_q
                        error = abs(q_value - target).item()
                else:
                    error = None
                
                self.memory.add(state, action, shaped_reward, next_state, done, error=error)

                # Multiple training iterations with adaptive frequency
                for _ in range(effective_train_per_step):
                    self.train()

                state = next_state
                total_reward += shaped_reward
                raw_total_reward += skip_reward
                self.env_count += 1
                step_count += 1
                
                # Save checkpoints at milestone steps with sanity check
                if self.env_count in self.milestone_checkpoints and not self.milestone_checkpoints[self.env_count]:
                    model_path = os.path.join(self.save_dir, f"model_steps_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved milestone checkpoint at {self.env_count} steps to {model_path}")
                    self.milestone_checkpoints[self.env_count] = True
                    
                    # Quick evaluation at milestone to verify quality
                    milestone_eval = self.evaluate(runs=3)
                    print(f"Milestone {self.env_count} quick eval: {milestone_eval:.2f}")
                    wandb.log({
                        f"Milestone_{self.env_count}_Eval": milestone_eval,
                        "Env Step Count": self.env_count
                    })

                # Less frequent logging to reduce overhead
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} AvgR: {self.avg_reward:.2f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Memory Size": len(self.memory),
                        "Last Reward": shaped_reward,
                        "Current Episode Reward": total_reward,
                        "Effective Train Per Step": effective_train_per_step
                    })
            
            # Update recent rewards for tracking progress
            self.recent_rewards.append(raw_total_reward)
            self.avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                    
            episode_duration = time.time() - episode_start_time
            print(f"[Eval] Ep: {ep} Total Reward: {raw_total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} AvgR: {self.avg_reward:.2f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": raw_total_reward,
                "Shaped Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Episode Length": step_count,
                "Episode Duration (s)": episode_duration,
                "Average Recent Reward": self.avg_reward
            })
            
            # Regular model checkpoints
            if ep % 5 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            # More thorough evaluation every 5 episodes
            if ep % 5 == 0:
                total_eval_reward = 0
                n_eval_runs = 10  # More runs for more reliable evaluation
                for i in range(n_eval_runs):
                    eval_reward = self.evaluate()
                    total_eval_reward += eval_reward
                eval_reward = total_eval_reward / n_eval_runs
                
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
                
                # Adaptive hyperparameter adjustment based on evaluation results
                # if ep > 10:
                #     # If we're not making progress, try adjusting hyperparameters
                #     if eval_reward < -15 and self.env_count > 200000:
                #         # Increase exploration
                #         self.epsilon = min(0.5, self.epsilon * 1.2)
                #         # Increase learning rate
                #         for param_group in self.optimizer.param_groups:
                #             param_group['lr'] = min(0.001, param_group['lr'] * 1.2)
                #         print(f"Adjusting hyperparameters: epsilon={self.epsilon:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")
                #     elif eval_reward > 10:
                #         # Reduce exploration once we're performing well
                #         self.epsilon = max(0.05, self.epsilon * 0.9)
                #         print(f"Reducing epsilon to {self.epsilon:.4f} based on good performance")

    def _shape_reward(self, reward):
        """Apply enhanced reward shaping to encourage better learning"""
        if reward > 0:
            return reward*2.5  # Stronger positive reward for scoring
        elif reward < 0:
            return reward*2  # Maintain negative reward for opponent scoring
        return -0.1
            
    def evaluate(self, runs=1):
        """Evaluates the current policy"""
        total_eval_reward = 0
        
        for _ in range(runs):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            episode_reward = 0

            while not done:
                # Always use greedy policy during evaluation
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()

                # Frame skipping during evaluation - match with training
                skip_frames = 2
                skip_reward = 0
                last_obs = None

                for i in range(skip_frames):
                    next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                    last_obs = next_obs
                    skip_reward += reward
                    done = terminated or truncated
                    if done:
                        break

                episode_reward += skip_reward
                state = self.preprocessor.step(last_obs)

            total_eval_reward += episode_reward
            
        return total_eval_reward / runs

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
            
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                
        self.train_count += 1
        
        # Adaptive batch size - increase batch size as we collect more experience
        effective_batch_size = min(self.batch_size, max(32, len(self.memory) // 500))
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(effective_batch_size)

        # Convert to tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values with Double Q-learning
        with torch.no_grad():
            # Use target network to select actions for more stable learning
            next_q_online = self.q_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # Add random noise to target values to prevent overfitting
            if self.env_count < 500000:  # Only in earlier training
                noise_scale = max(0.01, 0.1 * (1 - self.env_count / 500000))
                noise = torch.randn_like(next_q_values) * noise_scale
                next_q_values = next_q_values + noise
            
            # Compute target Q-values with discount adjusted for n-step returns
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.memory.n_steps) * next_q_values
        
        # Compute TD errors for updating priorities
        with torch.no_grad():
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            
        # Dynamic scaling of TD errors based on training progress
        progress_scale = min(1.0, self.env_count / 1000000)  # Scale from 0.0 to 1.0
        power_factor = 0.6 + 0.4 * progress_scale  # From 0.6 to 1.0
        
        # Scale errors with progress-dependent power factor
        td_errors = np.power(td_errors + 1e-6, power_factor) 
        
        # Use a mixture of losses for better stability
        # MSE for small errors, Huber for large errors
        errors = q_values - target_q_values
        mse_loss = self.mse_criterion(q_values, target_q_values)
        huber_loss = self.huber_criterion(q_values, target_q_values)
        
        # Blend losses based on error magnitude
        error_threshold = 1.0
        large_error_mask = (torch.abs(errors) > error_threshold).float()
        combined_loss = (1 - large_error_mask) * mse_loss + large_error_mask * huber_loss
        # combined_loss = huber_loss
        
        # Apply importance sampling weights
        weighted_loss = (weights * combined_loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping with adaptive norm based on training progress
        max_norm = 10.0 if self.env_count < 500000 else 5.0
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=max_norm)
        
        self.optimizer.step()
        # self.scheduler.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        # Adaptive target network update frequency
        if self.avg_reward > -15:  # Once we're making progress
            # Gradually decrease target update frequency as performance improves
            self.current_target_update = max(
                self.min_target_update,
                int(self.initial_target_update * (1 - (self.avg_reward + 21) / 40))
            )
        
        # Update target network using adaptive frequency
        if self.train_count % self.current_target_update == 0:
            # Soft update for more stable learning
            tau = 0.05 if self.env_count < 500000 else 0.1  # Adaptive tau
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Log metrics less frequently to reduce overhead
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {weighted_loss.detach().item():.4f} Q mean: {q_values.mean().detach().item():.3f} std: {q_values.std().detach().item():.3f} TUF: {self.current_target_update}")
            wandb.log({
                "Loss": weighted_loss.detach().item(),
                "Q Value Mean": q_values.mean().detach().item(),
                "Q Value Std": q_values.std().detach().item(),
                "Target Q Value Mean": target_q_values.mean().detach().item(),
                "TD Error Mean": np.mean(td_errors),
                "Learning Rate": self.optimizer.param_groups[0]['lr'],
                "Beta": self.memory.beta,
                "Effective Batch Size": effective_batch_size,
                "Target Update Frequency": self.current_target_update
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/task3_7")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run_task3_7")
    parser.add_argument("--batch-size", type=int, default=256)  # Increased batch size
    parser.add_argument("--memory-size", type=int, default=150000)  # Larger memory for better diversity
    parser.add_argument("--lr", type=float, default=0.0001)  # Optimized learning rate
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)  # Optimized decay
    parser.add_argument("--epsilon-min", type=float, default=0.09)  # Allow more exploitation
    parser.add_argument("--target-update-frequency", type=int, default=500)  # More frequent target updates
    parser.add_argument("--replay-start-size", type=int, default=20000)  # More initial data
    parser.add_argument("--max-episode-steps", type=int, default=10000)  # Longer episodes
    parser.add_argument("--train-per-step", type=int, default=4)  # More training per step
    parser.add_argument("--n-steps", type=int, default=4)  # Optimized n-step returns
    parser.add_argument("--per-alpha", type=float, default=0.6)  # Prioritization exponent
    parser.add_argument("--per-beta", type=float, default=0.4)  # Start with lower beta
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong_task3_7", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()