import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from ..utils.utils import to_numpy, preprocess_obs, postprocess_obs


class ActionDistribution:
    def __init__(self, params):
        self.action_dim = action_dim = params.action_dim
        self.continuous_action = params.continuous_state
        self.env_name = params.env_params.env_name

        model_based_params = params.policy_params.model_based_params
        self.n_top_candidate = model_based_params.n_top_candidate

        n_horizon_step = model_based_params.n_horizon_step
        std_scale = model_based_params.std_scale
        device = params.device

        if self.continuous_action:
            mu = torch.zeros(n_horizon_step, action_dim, dtype=torch.float32, device=device)
            std = torch.ones(n_horizon_step, action_dim, dtype=torch.float32, device=device) * std_scale
            self.init_dist = Normal(mu, std)

            action_low, action_high = params.action_spec
            self.action_low_device = torch.tensor(action_low, dtype=torch.float32, device=device)
            self.action_high_device = torch.tensor(action_high, dtype=torch.float32, device=device)
        else:
            probs = torch.ones(n_horizon_step, action_dim, dtype=torch.float32, device=device)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.init_dist = Categorical(probs=probs)
            test_probs = torch.ones(n_horizon_step, action_dim, dtype=torch.float32, device=device)
            test_probs[:,:5] = 0
            self.test_init_dist = Categorical(probs=test_probs)

        self.dist = self.init_dist

    def reset(self, stage='train'):
        if stage == 'train':
            self.dist = self.init_dist
        elif stage == 'test':
            assert self.env_name == 'Chemical'
            self.dist = self.test_init_dist
        else:
            raise NotImplementedError

    def sample(self, shape):
        """
        :param shape: int or tuple
        :return: (*shape, n_horizon_step, action_dim) if self.continuous_action else (*shape, n_horizon_step, 1)
        """
        if isinstance(shape, int):
            shape = (shape,)
        actions = self.dist.sample(shape)
        if self.continuous_action:
            actions = self.postprocess_action(actions)
        else:
            actions = actions.unsqueeze(dim=-1)
        return actions

    def update(self, actions, rewards):
        """
        :param actions: (n_candidate, n_horizon_step, action_dim) if self.continuous_action
            else (n_candidate, n_horizon_step, 1)
        :param rewards: (n_candidate, n_horizon_step, 1)
        :return:
        """
        sum_rewards = rewards.sum(dim=(1, 2))                           # (n_candidate,)

        top_candidate_idxes = torch.argsort(-sum_rewards)[:self.n_top_candidate]
        top_actions = actions[top_candidate_idxes]                      # (n_top_candidate, n_horizon_step, action_dim)

        if self.continuous_action:
            mu = top_actions.mean(dim=0)                                # (n_horizon_step, action_dim)
            std = torch.std(top_actions - mu, dim=0, unbiased=False)    # (n_horizon_step, action_dim)
            std = torch.clip(std, min=1e-6)
            self.dist = Normal(mu, std)
        else:
            top_actions = top_actions.squeeze(dim=-1)                   # (n_top_candidate, n_horizon_step)
            top_actions = F.one_hot(top_actions, self.action_dim)       # (n_top_candidate, n_horizon_step, action_dim)
            probs = top_actions.sum(dim=0)                              # (n_horizon_step, action_dim)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.dist = Categorical(probs=probs)

    def get_action(self):
        if self.continuous_action:
            action = self.dist.mean[0]
            action = self.postprocess_action(action)
        else:
            action = self.dist.probs[0].argmax()
        return to_numpy(action)

    @staticmethod
    def clip(val, min_val, max_val):
        return torch.min(torch.max(val, min_val), max_val)

    def postprocess_action(self, action):
        return self.clip(action, self.action_low_device, self.action_high_device)


class ModelBased(nn.Module):
    def __init__(self, encoder, inference, params):
        super(ModelBased, self).__init__()

        self.encoder = encoder
        self.inference = inference

        self.params = params
        self.device = device = params.device
        self.model_based_params = model_based_params = params.policy_params.model_based_params

        self.init_model()
        self.num_env = num_env = params.env_params.num_env
        self.model_reward_pred = None
        if num_env == 1:
            self.action_dist = ActionDistribution(params)
        elif num_env >= 2:
            self.action_dist = [ActionDistribution(params) for _ in range(num_env)]
        else:
            raise ValueError("num_env must be >= 1")

        if self.continuous_state:
            self.action_low, self.action_high = params.action_spec
            self.action_mean = (self.action_low + self.action_high) / 2
            self.action_scale = (self.action_high - self.action_low) / 2

        self.n_horizon_step = model_based_params.n_horizon_step
        self.n_iter = model_based_params.n_iter
        self.n_candidate = model_based_params.n_candidate

        self.to(device)
        self.optimizer = optim.Adam(self.fcs.parameters(), lr=params.policy_params.lr)

        self.load(params.training_params.load_model_based, device)
        self.train()

    def init_model(self):
        params = self.params
        model_based_params = self.model_based_params

        self.continuous_state = continuous_state = params.continuous_state

        feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.params.feature_inner_dim
        if not continuous_state:
            feature_dim = np.sum(feature_inner_dim)

        self.action_dim = action_dim = params.action_dim

        self.goal_keys = params.goal_keys
        obs_spec = params.obs_spec
        for key in self.goal_keys:
            assert obs_spec[key].ndim == 1, "Cannot concatenate because goal key {} is not 1D".format(key)
        goal_dim = np.sum([len(obs_spec[key]) for key in self.goal_keys])

        self.goal_inner_dim = None
        if not continuous_state:
            self.goal_inner_dim = []
            if self.goal_keys:
                self.goal_inner_dim = np.concatenate([params.obs_dims[key] for key in self.goal_keys])
            goal_dim = np.sum(self.goal_inner_dim)

        goal_dim = goal_dim.astype(np.int32)

        in_dim = feature_dim + action_dim + goal_dim
        modules = []
        for out_dim, activation in zip(model_based_params.fc_dims, model_based_params.activations):
            modules.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                activation = nn.ReLU()
            elif activation == "leaky_relu":
                activation = nn.LeakyReLU()
            elif activation == "tanh":
                activation = nn.Tanh()
            else:
                raise ValueError("Unknown activation: {}".format(activation))
            modules.append(activation)
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))

        self.fcs = nn.Sequential(*modules)

    def update_target(self,):
        pass

    def act_randomly(self):
        if self.continuous_state:
            return self.action_mean + self.action_scale * (np.random.uniform(-1, 1, self.action_scale.shape) if self.num_env == 1 else np.random.uniform(-1, 1, (self.num_env, *self.action_scale.shape)))
        else:
            return np.random.randint(self.action_dim) if self.num_env == 1 else np.random.randint(self.action_dim, size=self.num_env)

    def extract_goal_feature(self, obs):
        if not self.goal_keys:
            return None

        goal = torch.cat([obs[k] for k in self.goal_keys], dim=-1)
        if self.continuous_state:
            return goal
        else:
            goal = torch.unbind(goal, dim=-1)
            goal = [F.one_hot(goal_i.long(), goal_i_dim).float() if goal_i_dim > 1 else goal_i.unsqueeze(dim=-1)
                    for goal_i, goal_i_dim in zip(goal, self.goal_inner_dim)]
            return torch.cat(goal, dim=-1)

    def ground_truth_reward(self, feature, action, goal_feature):
        if not self.continuous_state:
            feature = torch.cat(feature, dim=-1)

        env_name = self.params.env_params.env_name
        if env_name == "Chemical":
            use_position = self.params.env_params.chemical_env_params.use_position
            current_color = []
            target_color = []
            idx = 0
            if use_position:
                for i, feature_inner_dim_i in enumerate(self.feature_inner_dim):
                    if i % 3 == 0:
                        current_color.append(feature[..., idx:idx + feature_inner_dim_i])
                    idx += feature_inner_dim_i
            else:
                for i, feature_inner_dim_i in enumerate(self.feature_inner_dim):
                    current_color.append(feature[..., idx:idx + feature_inner_dim_i])
                    idx += feature_inner_dim_i

            idx = 0
            for i, goal_inner_dim_i in enumerate(self.goal_inner_dim):
                target_color.append(goal_feature[..., idx:idx + goal_inner_dim_i])
                idx += goal_inner_dim_i

            num_matches = 0
            if self.encoder.chemical_train:
                match_type = self.encoder.chemical_match_type_train
            else:
                match_type = self.encoder.chemical_match_type_test
            for i, (current_color_i, target_color_i) in enumerate(zip(current_color, target_color)):
                if i not in match_type:
                    continue
                match = (current_color_i == target_color_i).all(dim=-1, keepdim=True)
                num_matches = num_matches + match
            pred_reward = num_matches
        else:
            raise NotImplementedError

        return pred_reward

    def act(self, obs, deterministic=False, stage='train'):
        """
        :param obs: (obs_spec)
        """
        if not deterministic and not self.continuous_state:
            if np.random.rand() < self.model_based_params.action_noise_eps:
                action = self.act_randomly()
                return action

        self.inference.eval()
        if isinstance(self.action_dist, list):
            for action_dist in self.action_dist:
                action_dist.reset(stage=stage)
        else:
            self.action_dist.reset(stage=stage)

        planner_type = self.model_based_params.planner_type
        if planner_type == "cem":
            action = self.cem(obs)
        else:
            raise ValueError("Unknown planner type: {}".format(planner_type))

        if not deterministic and self.continuous_state:
            action_noise = self.model_based_params.action_noise
            action_noise = np.random.normal(scale=action_noise, size=self.action_dim)
            action = np.clip(action + action_noise, self.action_low, self.action_high)

        if self.continuous_state:
            action = np.clip(action, self.action_low, self.action_high)
        # if stage == 'test' and self.params.env_params.env_name == 'Chemical':
        #     assert (action > 4).all()
        return action

    def get_abstraction_feature(self, feature):
        return feature

    def repeat_feature(self, feature, shape):
        """
        :param feature: 1-dimensional state/goal feature or None (do nothing if it's None)
        :param shape: repeat shape
        :return:
        """
        if feature is None:
            return None
        if isinstance(shape, int):
            shape = [shape]
        if isinstance(feature, torch.Tensor):
            return feature.expand(*shape, *feature.shape)
        else:
            return [feature_i.expand(*shape, *feature_i.shape) for feature_i in feature]

    def concat_current_and_next_features(self, feature, next_feature):
        feature = self.get_abstraction_feature(feature)                 # (n_candidate, feature_dim)
        next_feature = next_feature                                     # (n_candidate, n_horizon_step - 1, feature_dim)
        if self.continuous_state:
            # (n_candidate, n_horizon_step, feature_dim)
            return torch.cat([feature[:, None], next_feature], dim=1)   # (n_candidate, n_horizon_step, feature_dim)
        else:
            return [torch.cat([feature_i[:, None], next_feature_i], dim=1)
                    for feature_i, next_feature_i in zip(feature, next_feature)]

    def cem(self, obs):
        # cross-entropy method
        n_candidate = self.n_candidate
        inference = self.inference

        with torch.no_grad():
            raw_obs = obs
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            feature = self.encoder(obs)

            # assumed the goal is fixed in the episode
            goal_feature = self.extract_goal_feature(obs)

            feature = self.repeat_feature(feature, n_candidate)
            goal_feature = self.repeat_feature(goal_feature, (n_candidate, self.n_horizon_step))
            if self.num_env >= 2:
                if isinstance(feature, list):
                    feature = [feature_i.permute(1, 0, 2).reshape(n_candidate * self.num_env, -1)
                               for feature_i in feature]
                else:
                    feature = feature.permute(1, 0, 2).reshape(n_candidate * self.num_env, -1)
                
                if goal_feature is not None:
                    goal_feature = goal_feature.permute(2, 0, 1, 3).reshape(n_candidate * self.num_env, self.n_horizon_step, -1)

            iter_mean_history = {f"t{t}, d{d}": [] for t in range(self.n_horizon_step) for d in range(self.action_dim)}
            iter_std_history = {f"t{t}, d{d}": [] for t in range(self.n_horizon_step) for d in range(self.action_dim)}
            for i in range(self.n_iter):
                if isinstance(self.action_dist, list):
                    actions = torch.cat([action_dist.sample(n_candidate) for action_dist in self.action_dist], dim=0)
                else:
                    actions = self.action_dist.sample(n_candidate)  # (n_candidate, n_horizon_step, action_dim)

                # (n_candidate, n_horizon_step, 1)
                pred_next_dist = inference.forward_with_feature(feature, actions)
                pred_next_feature = inference.sample_from_distribution(pred_next_dist)
                pred_rewards = self.ground_truth_reward(pred_next_feature, actions, goal_feature)
                if self.num_env == 1:
                    self.action_dist.update(actions, pred_rewards)
                else:
                    pred_rewards = pred_rewards.reshape(self.num_env, n_candidate, self.n_horizon_step, 1)
                    if self.continuous_state:
                        actions = actions.reshape(self.num_env, n_candidate, self.n_horizon_step, self.action_dim)
                    else:
                        actions = actions.reshape(self.num_env, n_candidate, self.n_horizon_step, 1)
                    for i in range(self.num_env):
                        self.action_dist[i].update(actions[i], pred_rewards[i])

                # Log the mean and std of the action distribution
                if self.params.continuous_action:
                    if self.params.stage == 'train' and (self.params.step + 1) % self.params.training_params.plot_freq == 0:
                        if self.num_env == 1:
                            dist = self.action_dist.dist
                        else:
                            dist = self.action_dist[0].dist
                        for t in range(self.n_horizon_step):
                            for d in range(self.action_dim):
                                iter_mean_history[f"t{t}, d{d}"].append(dist.mean[t, d].item())
                                iter_std_history[f"t{t}, d{d}"].append(dist.stddev[t, d].item())
            
        if self.num_env == 1:
            action = self.action_dist.get_action()
        else:
            action = np.stack([action_dist.get_action() for action_dist in self.action_dist])

        return action

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "fcs" in k}
        own_state.update(state_dict)
        self.load_state_dict(own_state)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("ModelBased loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_my_state_dict(checkpoint["model"])                # only load reward predictor
            self.optimizer.load_state_dict(checkpoint["optimizer"])
