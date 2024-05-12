import numpy as np

import torch
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical

from .inference import Inference


class InferenceMLP(Inference):
    def __init__(self, encoder, params):
        self.mlp_params = params.inference_params.mlp_params
        super(InferenceMLP, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        mlp_params = self.mlp_params

        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.params.feature_inner_dim
        self.num_action_variable = params.num_action_variable

        if continuous_state:
            in_dim = feature_dim + action_dim
            if self.learn_std:
                final_dim = 2 * feature_dim
            else:
                final_dim = feature_dim
        else:
            in_dim = np.sum(feature_inner_dim) + action_dim
            final_dim = np.sum(feature_inner_dim) + np.sum(feature_inner_dim == 1)

        fcs = nn.Sequential()
        for idx, out_dim in enumerate(mlp_params.fc_dims):
            fcs.add_module(f"fc_{idx}", nn.Linear(in_dim, out_dim))
            fcs.add_module(f"relu_{idx}", nn.ReLU())
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, final_dim))

        self.fcs = fcs

    def forward_step(self, feature, action):

        if not self.continuous_state:
            feature_copy = feature
            feature = torch.cat(feature, dim=-1)

        inputs = torch.cat([feature, action], dim=-1)
        dist = self.fcs(inputs)

        if self.continuous_state:
            if self.learn_std:
                mu, log_std = torch.split(dist, int(self.feature_dim), dim=-1)
            else:
                mu = dist
                log_std = torch.zeros_like(mu)
            return self.normal_helper(mu, feature, log_std)
        else:
            split_sections = [2 if feature_i_inner_dim == 1 else feature_i_inner_dim
                              for feature_i_inner_dim in self.feature_inner_dim]
            raw_dist = torch.split(dist, split_sections, dim=-1)

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(feature_copy, self.feature_inner_dim, raw_dist):
                dist_i = dist_i.squeeze(dim=0)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist
