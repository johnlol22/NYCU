import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import racecar_gym
from racecar_gym.env import RaceEnv
import torch.nn as nn
import torch
from stable_baselines3.td3.policies import TD3Policy

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Calculate the features dimension first
        n_input_channels = observation_space.shape[0]
        self._features_dim = features_dim  # Store temporarily
        
        # Call super() with the features_dim
        super().__init__(observation_space, features_dim=self._features_dim)
        
        print(f"Creating CNN with {n_input_channels} input channels")
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, self._features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


scenario = 'circle_cw_competition_collisionStop'
env = RaceEnv(scenario=scenario,
                  render_mode='rgb_array_birds_eye',
                  reset_when_collision=True if 'austria' in scenario else False)
env = Monitor(env)
#eval_callback = EvalCallback(env, best_model_save_path='./td3_model/',
#                             log_path='./td3_racecar_tensorboard/TD3_9_eval', eval_freq=500,
#                             deterministic=True, render=False)

# The noise objects for TD3
n_actions = env.action_space.shape
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create model
policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": dict(features_dim=512),
    "net_arch": [256, 256],
    "normalize_images": False
}

model = TD3(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cuda",
    action_noise=action_noise,
    buffer_size=5000,
    learning_rate=3e-4,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    policy_delay=10,
    learning_starts=1000,
    target_policy_noise=0.2,  # Added: noise added to target actions
    target_noise_clip=0.5,  # Added: clipping of target noise
    tensorboard_log="./td3_racecar_tensorboard/"
)

model.learn(total_timesteps=3e7, log_interval=10)
model.save("./td3_10_for_circle.zip")
vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = TD3.load("td3_pendulum")

obs = vec_env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
