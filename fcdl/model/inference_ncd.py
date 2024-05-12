import torch
import torch.nn as nn
import torch.nn.functional as F

from .gumbel import GumbelMatrix_NCD
from .inference_ours_masking import InferenceOursMask
from .inference_utils import forward_network

EPS = 1e-4


class InferenceNCD(InferenceOursMask):
    def __init__(self, encoder, params):
        super(InferenceNCD, self).__init__(encoder, params)

    def init_model(self):
        super(InferenceNCD, self).init_model()

        ours_params = self.params.ours_params
        fc_dims = ours_params.feature_fc_dims
        
        feature_dim = self.feature_dim
        action_dim = self.action_dim

        self.learn_codebook = False
        
        if self.use_gt_global_mask:
            self.get_gt_global_mask(self.num_state_var, self.num_action_variable)
        else:
            self.local_causal_model = GumbelMatrix_NCD(self.params, feature_dim, action_dim, self.num_state_var, self.num_action_variable, self.continuous_state, fc_dims, self.device)
    
    def forward_with_local_mask(self, state_feature, action_feature, feature, local_mask, prob, current_pred_step):
        s_dim = state_feature.size(0)
        sampling_num = local_mask.size(0)

        original_sa_feature = torch.cat([state_feature, action_feature], dim=0)
        original_sa_feature = original_sa_feature.repeat(s_dim, 1, 1, 1)

        sampled_dist = []
        
        for i in range(sampling_num):
            
            if self.code_labeling:
                code_label = local_mask[i].detach()
                code_label = code_label.squeeze(-1)
                code_label = code_label.permute(0, 2, 1)
            sa_feature = original_sa_feature * local_mask[i]
            sa_feature = sa_feature.permute(0, 2, 1, 3)
            sa_feature = sa_feature.reshape(*sa_feature.shape[:2], -1)
            
            if self.code_labeling:
                sa_feature = torch.cat([sa_feature, code_label], dim=-1) 
                # p(y | x, z) in Eq. (3) in NCD paper
                # For the feature masking, labeling is optional in practice
                # For the input masking, labeling is necessary

            sa_feature = forward_network(sa_feature, self.sa_feature_weights, self.sa_feature_biases)
            sa_feature = F.relu(sa_feature)

            sampled_dist.append(self.predict_from_sa_feature(sa_feature, feature))
        
        mean_dist = self.mean_dist(sampled_dist)

        return mean_dist

    def update(self, obses, actions, next_obses, eval=False):
        self.is_eval = eval
        assert not self.training == self.is_eval
        features = self.encoder(obses)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(features, actions)

        masked_pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)
        masked_pred_loss = masked_pred_loss.mean()
        reg_loss = self.local_causal_model.total_loss()
        loss = masked_pred_loss + reg_loss
        loss_detail = {"pred_loss": masked_pred_loss,
                       "reg_loss": self.local_causal_model.reg_loss.mean(),
        }

        if not eval:
            self.backprop(loss, loss_detail)

        return loss_detail

    def train(self, training=True):
        self.training = training
        if training:
            self.local_causal_model.train()
            self.local_causal_model.training = True
        else:
            self.local_causal_model.eval()
            self.local_causal_model.training = False
    
    def eval_prediction(self, obs, actions, next_obses, info_batch):
        self.training = False
        self.eval()
        self.local_causal_model.training = False
        self.local_causal_model.eval()
        return InferenceOursMask.eval_prediction(self, obs, actions, next_obses, info_batch)
