import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 1)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, a=None):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)

        value = self.value(x)       #state value function ?
        value = torch.squeeze(value) #remove the dimension value is 1 , resulting a scalar

        logits = self.action_logits(x)

        dist = Categorical(logits=logits) #logits are the unnormalize(not sum to 1) log probability of actions, categorical is a type
    
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy
        '''
        high entropy means the agent is exploring the actions, it helps the agent discover new strategies and avoid getting stuck in suboptimal solutions

        low entropy means the policy is more deterministic, the agent is exploiting its knowledge
        '''
        if eval:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action  = dist.sample()
        if a is not None:
            action_probability = dist.log_prob(a)
        else:
            action_probability = dist.log_prob(action)
        entropy = dist.entropy()
        return action, action_probability, value, entropy
        #return NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                


