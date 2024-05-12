import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])

        self.continuous_state = params.continuous_state
        self.feature_inner_dim = params.feature_inner_dim

        self.chemical_train = True
        self.chemical_match_type_train = list(range(len(self.keys)))
        self.to(params.device)
    
    def get_clean_obs(self, obs, detach=False):
        if self.continuous_state:
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
        return obs

    def forward(self, obs, detach=False):
        if self.continuous_state:
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]

            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                test_level = self.chemical_test_level
                if test_level == 0:
                    noise_variable_list = [1, 2] # number of noisy nodes = 2
                elif test_level == 1:
                    noise_variable_list = [1, 2, 3, 4] # number of noisy nodes = 4
                elif test_level == 2:
                    noise_variable_list = [1, 2, 3, 4, 5, 6] # number of noisy nodes = 6
                self.chemical_match_type_test = list(set(self.chemical_match_type_train) - set(noise_variable_list))
                
                if test_scale == 0:
                    return obs
                else:
                    for i in noise_variable_list:
                        obs[i] = obs[i] * torch.randn_like(obs[i]) * test_scale
                    return obs
            else: return obs


_AVAILABLE_ENCODERS = {"identity": IdentityEncoder}


def make_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    return _AVAILABLE_ENCODERS[encoder_type](params)
