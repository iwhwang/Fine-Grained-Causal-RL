import numpy as np
import torch.nn as nn

class RandomPolicy(nn.Module):
    def __init__(self, params):
        super(RandomPolicy, self).__init__()
        self.params = params
        self.continuous_action = params.continuous_action
        self.num_env = params.env_params.num_env
        if self.continuous_action:
            action_low, action_high = params.action_spec
            self.action_low = action_low
            self.action_high = action_high
            self.action_mean = (action_low + action_high) / 2
            self.action_scale = (action_high - action_low) / 2
        else:
            self.action_dim = params.action_dim

    def act_randomly(self):
        if self.continuous_action:
            return self.action_mean + self.action_scale * (np.random.uniform(-1, 1, self.action_scale.shape) if self.num_env == 1 else np.random.uniform(-1, 1, (self.num_env, *self.action_scale.shape)))
        else:
            return np.random.randint(self.action_dim) if self.num_env == 1 else np.random.randint(self.action_dim, size=self.num_env)

    def act(self, obs):
        action = self.act_randomly()
        if self.continuous_action:
            action = np.clip(action, self.action_low, self.action_high)
        return action

    def save(self, path):
        pass
