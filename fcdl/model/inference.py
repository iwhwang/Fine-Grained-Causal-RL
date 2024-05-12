import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.kl import kl_divergence

from ..utils.utils import to_numpy, preprocess_obs, postprocess_obs


class Inference(nn.Module):
    def __init__(self, encoder, params):
        super(Inference, self).__init__()

        self.encoder = encoder

        self.params = params
        self.device = device = params.device
        self.inference_params = inference_params = params.inference_params
        self.algo_name = params.training_params.inference_algo

        self.residual = inference_params.residual
        self.log_std_min = inference_params.log_std_min
        self.log_std_max = inference_params.log_std_max
        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action
        self.num_action_variable = params.num_action_variable
        self.learn_std = inference_params.learn_std
        self.env_name = params.env_params.env_name
        self.use_gt_global_mask = params.inference_params.use_gt_global_mask
        self.num_action_variable = params.num_action_variable

        self.init_model()
        self.reset_params()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=inference_params.lr)

        self.load(params.training_params.load_inference, device)
        self.train()
    
    def get_gt_global_mask(self, num_state_variable, num_action_variable):
        if self.env_name == 'Chemical':
            self.gt_global_mask = torch.zeros(num_state_variable, num_state_variable + num_action_variable).to(self.device)
            self.gt_global_mask.fill_diagonal_(1)
            self.gt_global_mask[:, -1] = 1
            lower_indices = np.tril_indices(num_state_variable)
            self.gt_global_mask[lower_indices[0], lower_indices[1]] = 1
        else:
            raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def reset_params(self):
        pass

    def forward_step(self, feature, action):
        raise NotImplementedError

    def mean_dist(self, dist_list):
        if self.continuous_state: 
            n_sampling = len(dist_list)
            mu = torch.stack([dist.mean for dist in dist_list], dim=-2)
            mean_mu = mu.mean(dim=1)
            if self.learn_std:
                std = torch.stack([dist.stddev for dist in dist_list], dim=-2)
                mean_std = torch.norm(std, 2, 1) / n_sampling
            else:
                mean_std = torch.ones_like(mean_mu)
            return Normal(mean_mu, mean_std)
        else:
            mean_dist = []
            for i, dist_i in enumerate(dist_list[0]):
                if isinstance(dist_i, Normal): 
                    mu = torch.stack([dist[i].mean for dist in dist_list], dim=-2)
                    std = torch.stack([dist[i].stddev for dist in dist_list], dim=-2)
                    mean_dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    probs = torch.stack([dist[i].probs for dist in dist_list], dim=-2)
                    probs = probs.mean(dim=1)
                    mean_dist_i = OneHotCategorical(probs=probs)
                else:
                    raise NotImplementedError
                mean_dist.append(mean_dist_i)

            return mean_dist

    def stack_dist(self, dist_list):
        if self.continuous_state:
            mu = torch.stack([dist.mean for dist in dist_list], dim=-2)
            std = torch.stack([dist.stddev for dist in dist_list], dim=-2)
            return Normal(mu, std)
        else:
            stacked_dist_list = []
            for i, dist_i in enumerate(dist_list[0]):
                if isinstance(dist_i, Normal):
                    mu = torch.stack([dist[i].mean for dist in dist_list], dim=-2)
                    std = torch.stack([dist[i].stddev for dist in dist_list], dim=-2)
                    stacked_dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    logits = torch.stack([dist[i].logits for dist in dist_list], dim=-2)
                    stacked_dist_i = OneHotCategorical(logits=logits)
                else:
                    raise NotImplementedError
                stacked_dist_list.append(stacked_dist_i)

            return stacked_dist_list

    def normal_helper(self, mean_, base_, log_std_):
        if self.residual:
            mean_ = mean_ + base_
        if self.learn_std:
            log_std_ = torch.clip(log_std_, min=self.log_std_min, max=self.log_std_max)
            std_ = torch.exp(log_std_)
        else:
            std_ = torch.ones_like(mean_)
        return Normal(mean_, std_)

    def sample_from_distribution(self, dist):
        if self.continuous_state:
            return dist.rsample() if (self.training and self.learn_std) else dist.mean
        else:
            sample = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    sample_i = dist_i.rsample() if (self.training and self.learn_std) else dist_i.mean
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits
                    if self.training:
                        sample_i = F.gumbel_softmax(logits, hard=True)
                    else:
                        sample_i = F.one_hot(torch.argmax(logits, dim=-1), logits.size(-1)).float()
                else:
                    raise NotImplementedError
                sample.append(sample_i)
            return sample

    def log_prob_from_distribution(self, dist, value):
        if self.continuous_state:
            if self.learn_std:
                return dist.log_prob(value)
            else:
                return -(dist.mean - value)**2
        else:
            log_prob = []
            for dist_i, val_i in zip(dist, value):
                if isinstance(dist_i, Normal):
                    if not self.learn_std:
                        new_dist_i = Normal(dist_i.mean, torch.ones_like(dist_i.stddev))
                        dist_i = new_dist_i
                    log_prob_i = dist_i.log_prob(val_i)
                    log_prob_i = log_prob_i.squeeze(dim=-1)
                else:
                    log_prob_i = dist_i.log_prob(val_i)
                log_prob.append(log_prob_i)
            return torch.stack(log_prob, dim=-1)

    def forward_with_feature(self, feature, actions, **kwargs):        
        if not self.continuous_action:
            actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()
        actions = torch.unbind(actions, dim=-2)
        dists = []
        for action in actions:
            dist = self.forward_step(feature, action)

            feature = self.sample_from_distribution(dist)
            dists.append(dist)
        dists = self.stack_dist(dists)

        return dists

    def get_feature(self, obs):
        feature = self.encoder(obs)
        if isinstance(feature, Distribution):
            assert isinstance(feature, Normal)
            feature = feature.mean
        return feature

    def forward(self, obs, actions):
        feature = self.get_feature(obs)
        return self.forward_with_feature(feature, actions)

    def prediction_loss_from_dist(self, pred_dist, next_feature, keep_variable_dim=False):
        if isinstance(next_feature, Distribution):
            assert isinstance(next_feature, Normal)
            next_feature = Normal(next_feature.mean.detach(), next_feature.stddev.detach())
            pred_loss = kl_divergence(next_feature, pred_dist)
        else:
            if self.continuous_state:
                next_feature = next_feature.detach()
            else:
                next_feature = [next_feature_i.detach() for next_feature_i in next_feature]
            pred_loss = -self.log_prob_from_distribution(pred_dist, next_feature)
        
        if not keep_variable_dim:
            pred_loss = pred_loss.mean(dim=-1)

        return pred_loss

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.inference_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        self.optimizer.step()
        return loss_detail

    def update(self, obses, actions, next_obses, eval=False):
        assert not self.training == eval
        features = self.encoder(obses)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(features, actions)

        pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)
        loss = pred_loss = pred_loss.mean()
        loss_detail = {"pred_loss": pred_loss}

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def preprocess(self, obs, actions, next_obses):
        if isinstance(actions, int):
            actions = np.array([actions])

        if isinstance(actions, np.ndarray):
            if self.continuous_action and actions.dtype != np.float32:
                actions = actions.astype(np.float32)
            if not self.continuous_action and actions.dtype != np.int64:
                actions = actions.astype(np.int64)
            actions = torch.from_numpy(actions).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            next_obses = postprocess_obs(preprocess_obs(next_obses, self.params))
            next_obses = {k: torch.from_numpy(v).to(self.device) for k, v in next_obses.items()}

        need_squeeze = False
        if actions.ndim == 1:
            need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}
            actions = actions[None, None]
            next_obses = {k: v[None, None] for k, v in next_obses.items()}
        elif self.params.env_params.num_env > 1 and actions.ndim == 2:
            need_squeeze = True
            actions = actions[:, None]
            next_obses = {k: v[:, None] for k, v in next_obses.items()}

        return obs, actions, next_obses, need_squeeze

    def ood_prediction(self, obs, actions, next_obses, info_batch):
        self.is_eval = eval
        assert not self.training
        self.eval()
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)
        if len(actions.shape) == 2: actions = actions.unsqueeze(-1)
        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder.get_clean_obs(next_obses)

            pred_next_dist = self.forward_with_feature(feature, actions)
            pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature, keep_variable_dim=True)
            loss_detail = {}

            if self.params.env_params.env_name == "Chemical":
                if self.encoder.chemical_train:
                    match_type = self.encoder.chemical_match_type_train
                else:
                    match_type = self.encoder.chemical_match_type_test
                accuracy = []
                for i, (dist_i, next_feature_i) in enumerate(zip(pred_next_dist, next_feature)):
                    if not isinstance(dist_i, OneHotCategorical):
                        continue
                    if not i in match_type:
                        continue
                    logits = dist_i.logits
                    accuracy_i = logits.argmax(dim=-1) == next_feature_i.argmax(dim=-1)
                    accuracy.append(accuracy_i)
                accuracy = torch.stack(accuracy, dim=-1)
                accuracy = to_numpy(accuracy)
                loss_detail["accuracy"] = accuracy.mean()
            else:
                raise NotImplementedError
        return loss_detail

    def eval_prediction(self, obs, actions, next_obses, info_batch):
        self.is_eval = eval
        assert not self.training == self.is_eval
        self.eval()
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)
        if len(actions.shape) == 2: actions = actions.unsqueeze(-1)

        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obses)
            pred_next_dist = self.forward_with_feature(feature, actions)

            pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature, keep_variable_dim=True)
            loss_detail = {"pred_loss": pred_loss.mean()}

            if self.params.env_params.env_name == "Chemical":
                accuracy = None

                if not self.continuous_state:
                    accuracy = []
                    for dist_i, next_feature_i in zip(pred_next_dist, next_feature):
                        if not isinstance(dist_i, OneHotCategorical):
                            continue
                        logits = dist_i.logits
                        accuracy_i = logits.argmax(dim=-1) == next_feature_i.argmax(dim=-1)
                        accuracy.append(accuracy_i)
                    accuracy = torch.stack(accuracy, dim=-1)
                    accuracy = to_numpy(accuracy)
                    loss_detail["accuracy"] = accuracy.mean()
        return loss_detail

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("inference loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])