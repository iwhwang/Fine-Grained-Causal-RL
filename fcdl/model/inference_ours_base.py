from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

from .gumbel import VQVAEGumbelMatrixLatent
from .inference import Inference
from .inference_utils import forward_network, forward_network_batch, reset_layer


class InferenceOursBase(Inference, metaclass=ABCMeta):
    def __init__(self, encoder, params):
        self.is_eval = True
        super(InferenceOursBase, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        ours_params = self.params.ours_params

        # model params
        continuous_state = self.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = self.params.feature_inner_dim

        device = self.device
        self.local_mask_sampling_num = params.ours_params.local_mask_sampling_num
        self.eval_local_mask_sampling_num = params.ours_params.eval_local_mask_sampling_num
        self.code_labeling = ours_params.code_labeling
        self.learn_codebook = not self.use_gt_global_mask

        fc_dims = ours_params.feature_fc_dims

        num_state_var = feature_dim
        self.num_state_var = num_state_var
        if self.use_gt_global_mask:
            self.get_gt_global_mask(self.num_state_var, self.num_action_variable)
        else:
            self.local_causal_model = VQVAEGumbelMatrixLatent(self.params, feature_dim, action_dim, num_state_var, self.num_action_variable, self.continuous_state, fc_dims, device)

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.sa_feature_weights = nn.ParameterList() 
        self.sa_feature_biases = nn.ParameterList()
        self.generative_weights = nn.ParameterList()
        self.generative_biases = nn.ParameterList()

        self.state_feature_1st_layer_weights = nn.ParameterList()
        self.state_feature_1st_layer_biases = nn.ParameterList()
        self.generative_last_layer_weights = nn.ParameterList()
        self.generative_last_layer_biases = nn.ParameterList()

        if self.num_action_variable == 1:
            in_dim = action_dim
        else:
            in_dim = 1
        for out_dim in ours_params.feature_fc_dims[:1]:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(self.num_action_variable, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(self.num_action_variable, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            out_dim = ours_params.feature_fc_dims[0]
            self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
        else:
            out_dim = ours_params.feature_fc_dims[0]
            for feature_i_dim in self.feature_inner_dim:
                in_dim = feature_i_dim
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))

        fc_dims = ours_params.feature_fc_dims[1:]
        in_dim = (self.num_state_var + self.num_action_variable) * out_dim
        if self.params.ours_params.code_labeling:
            if 'ours' in self.params.training_params.inference_algo:
                in_dim = in_dim + self.params.ours_params.codebook_size
            elif 'ncd' in self.params.training_params.inference_algo:
                in_dim = in_dim + (self.num_state_var + self.num_action_variable)
        for out_dim in fc_dims:
            self.sa_feature_weights.append(nn.Parameter(torch.zeros(self.num_state_var, in_dim, out_dim)))
            self.sa_feature_biases.append(nn.Parameter(torch.zeros(self.num_state_var, 1, out_dim)))
            in_dim = out_dim
        
        in_dim = ours_params.feature_fc_dims[-1]
        for out_dim in ours_params.generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(self.num_state_var, in_dim, out_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(self.num_state_var, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            final_dim = 2 if self.learn_std else 1
            self.generative_weights.append(nn.Parameter(torch.zeros(self.num_state_var, in_dim, final_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(self.num_state_var, 1, final_dim)))
        else:
            for feature_i_dim in self.feature_inner_dim:
                final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        feature_dim = self.feature_dim
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(self.num_action_variable):
                reset_layer(w[i], b[i])
        temp = feature_dim if self.continuous_state else 1
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            for i in range(temp):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(self.num_state_var):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)
        
        self.reset_params_sa_feature()

    @abstractmethod
    def reset_params_sa_feature(self):
        ...

    def extract_action_feature(self, action):
        if self.num_action_variable == 1:
            action = action.unsqueeze(dim=0)
        else:
            action = action.permute(1, 0)
            action = action.unsqueeze(dim=-1)
        action_feature = forward_network(action, self.action_feature_weights, self.action_feature_biases)
        action_feature = F.relu(action_feature)
        return action_feature

    def extract_state_feature(self, feature):
        if self.continuous_state:
            x = feature.transpose(0, 1)
            x = x.unsqueeze(-1)
            x = forward_network(x, self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases)
            x = F.relu(x)
        else:
            reshaped_feature = []
            for f_i in feature:
                f_i = f_i.unsqueeze(0)
                reshaped_feature.append(f_i)
            x = forward_network_batch(reshaped_feature,
                                        self.state_feature_1st_layer_weights,
                                        self.state_feature_1st_layer_biases)
            x = torch.stack(x, dim=1)
            x = x.squeeze(0)

        return x

    def predict_from_sa_feature(self, sa_feature, residual_base=None):
        x = forward_network(sa_feature, self.generative_weights, self.generative_biases)

        if self.continuous_state:
            x = x.permute(1, 0, 2)
            if self.learn_std:
                mu, log_std = x.unbind(dim=-1)
            else:
                mu = x.squeeze(-1)
                log_std = torch.ones_like(mu)
            return self.normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]
            x = forward_network_batch(x,
                                        self.generative_last_layer_weights,
                                        self.generative_last_layer_biases,
                                        activation=None)

            feature_inner_dim = self.feature_inner_dim

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(residual_base, feature_inner_dim, x):
                dist_i = dist_i.squeeze(dim=0)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_with_feature(self, feature, actions, **kwargs):
        if not self.continuous_action:
            actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()
        actions = torch.unbind(actions, dim=-2)
        
        dists = []
        current_pred_step = 0
        for action in actions:
            dist = self.forward_step(feature, action, current_pred_step)
            feature = self.sample_from_distribution(dist)
            dists.append(dist)
            current_pred_step += 1
        dists = self.stack_dist(dists)

        return dists
    
    def forward_step(self, feature, action, current_pred_step):
        if self.training:
            sampling_num = self.local_mask_sampling_num
        else: 
            sampling_num = self.eval_local_mask_sampling_num
        
        action_feature = self.extract_action_feature(action)
        state_feature = self.extract_state_feature(feature)
        
        bs = state_feature.size(1)
        if self.use_gt_global_mask:
            global_mask = self.gt_global_mask.clone().repeat(bs, 1, 1)
            prob = global_mask
            global_mask = global_mask.repeat(sampling_num, 1, 1, 1)
            local_mask = global_mask
        else:
            local_mask, prob = self.local_causal_model(state_feature, action_feature, current_pred_step, training=self.training)
            if not self.training:
                prob = (prob > 0.5).float()
            prob = prob.detach()
        
        local_mask = local_mask.permute(0, 2, 3, 1)
        local_mask = local_mask.unsqueeze(dim=-1)
        prob = prob.permute(1, 2, 0)
        prob = prob.unsqueeze(dim=-1)
        assert sampling_num == local_mask.size(0)

        return self.forward_with_local_mask(state_feature, action_feature, feature, local_mask, prob, current_pred_step)

    @abstractmethod
    def forward_with_local_mask(self, state_feature, action_feature, feature, local_mask, prob, current_pred_step):
        ...

    def get_code_label(self, current_pred_step):
        code_label = self.local_causal_model.code_index[current_pred_step].clone().detach()
        return F.one_hot(code_label, self.local_causal_model.codebook_size)

    def eval_prediction(self, obs, actions, next_obses, info_batch):
        self.training = False
        self.eval()
        if self.learn_codebook:
            self.local_causal_model.training = False
            self.local_causal_model.emb.training = False
            self.local_causal_model.eval()
            self.local_causal_model.emb.eval()
        return Inference.eval_prediction(self, obs, actions, next_obses, info_batch)

    def forward(self, obs, actions):
        feature = self.encoder(obs)
        return self.forward_with_feature(feature, actions)

    def update(self, obses, actions, next_obses, eval=False):
        self.is_eval = eval
        assert not self.training == self.is_eval
        features = self.encoder(obses)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(features, actions)

        masked_pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)
        masked_pred_loss = masked_pred_loss.mean()
        full_pred_loss = torch.zeros_like(masked_pred_loss)
        loss = full_pred_loss + masked_pred_loss
        loss_detail = {"pred_loss": masked_pred_loss, 
                       "full_pred_loss": full_pred_loss
        }
        if self.learn_codebook:
            ours_loss = self.local_causal_model.total_loss()
            loss = loss + ours_loss
            loss_detail['reg_loss']= self.local_causal_model.reg_loss.mean()
            loss_detail['vq_loss']= self.local_causal_model.vq_loss.mean()
            loss_detail['commit_loss']= self.local_causal_model.commit_loss.mean()

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def train(self, training=True):
        self.training = training
        if self.learn_codebook:
            if training:
                self.local_causal_model.train()
                self.local_causal_model.emb.train()
                self.local_causal_model.training = True
                self.local_causal_model.emb.training = True
            else:
                self.local_causal_model.eval()
                self.local_causal_model.emb.eval()
                self.local_causal_model.training = False
                self.local_causal_model.emb.training = False

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, path)

    def load(self, path, device):
        Inference.load(self, path, device)
