import torch
import torch.nn.functional as F

from .inference_ours_base import InferenceOursBase
from .inference_utils import forward_network, reset_layer

class InferenceOursMask(InferenceOursBase):
    def __init__(self, encoder, params):
        super(InferenceOursMask, self).__init__(encoder, params)

    def init_model(self):
        super(InferenceOursMask, self).init_model()

    def reset_params_sa_feature(self):
        for w, b in zip(self.sa_feature_weights, self.sa_feature_biases):
            for i in range(self.num_state_var):
                reset_layer(w[i], b[i])
    
    def forward_with_local_mask(self, state_feature, action_feature, feature, local_mask, prob, current_pred_step):
        s_dim = state_feature.size(0)
        sampling_num = local_mask.size(0)

        original_sa_feature = torch.cat([state_feature, action_feature], dim=0)
        original_sa_feature = original_sa_feature.repeat(s_dim, 1, 1, 1)

        if self.code_labeling:
            code_label = self.get_code_label(current_pred_step)
            code_label = code_label.repeat(s_dim, 1, 1)

        sampled_dist = []
        
        for i in range(sampling_num):
            sa_feature = original_sa_feature * local_mask[i]
            
            sa_feature = sa_feature.permute(0, 2, 1, 3)
            sa_feature = sa_feature.reshape(*sa_feature.shape[:2], -1)
            
            if self.code_labeling:
                sa_feature = torch.cat([sa_feature, code_label], dim=-1)
                # Code labeling is optional in practice
                # See Appendix C.3.2 for the discussion on the design choices

            sa_feature = forward_network(sa_feature, self.sa_feature_weights, self.sa_feature_biases)
            sa_feature = F.relu(sa_feature)

            sampled_dist.append(self.predict_from_sa_feature(sa_feature, feature))
        
        mean_dist = self.mean_dist(sampled_dist)

        return mean_dist
