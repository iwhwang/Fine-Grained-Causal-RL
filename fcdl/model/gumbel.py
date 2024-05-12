import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .nearest_embed import NearestEmbed, NearestEmbedEMA


EPS = 1e-6

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=math.sqrt(3))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def sample_logistic(shape, device):
    u = torch.rand(shape, dtype=torch.float32, device=device)
    u = torch.clip(u, EPS, 1 - EPS)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid_batch(log_alpha, device, bs=None, tau=1, hard=False):
    if bs is None:
        shape = log_alpha.shape
    else:
        shape = tuple([bs] + list(log_alpha.size()))

    logistic_noise = sample_logistic(shape, device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def gumbel_sigmoid(log_alpha, device, bs=None, tau=1, hard=False):
    if bs is None or bs==1:
        shape = log_alpha.shape
    else:
        try: bs = bs[0]
        except: pass
        shape = tuple([bs] + list(log_alpha.size()))

    logistic_noise = sample_logistic(shape, device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

class VQVAEGumbelMatrixLatent(torch.nn.Module):
    def __init__(self, params, feature_dim, action_dim, num_state_var, num_action_var, continuous_state, fc_dims, device):
        super(VQVAEGumbelMatrixLatent, self).__init__()
        self.local_mask_sampling_num = params.ours_params.local_mask_sampling_num
        self.eval_local_mask_sampling_num = params.ours_params.eval_local_mask_sampling_num
        self.continuous_state = continuous_state
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.num_action_var = num_action_var
        self.num_state_var = num_state_var
        final_dim = feature_dim + 1
        self.final_dim = final_dim
        self.device = device
        self.fc_dims = fc_dims
        self.learn_action = params.inference_params.learn_action # Chemical: False, Magnetic: True
        self.learn_upper = params.inference_params.learn_upper # Chemical: False, Magnetic: True
        self.lcm_dim_1 = self.num_state_var
        self.lcm_dim_2 = self.num_state_var + self.num_action_var
        
        self.lcm_dim_2 = self.lcm_dim_2 - 1
        if not self.learn_action: self.lcm_dim_2 = self.lcm_dim_2 - self.num_action_var
        self.adjust_dimension = self.adjust_dimension_default
        self.input_dim = 0
        self.ours_type = params.training_params.inference_algo
        
        self.preprocess = self.preprocess_ours_mask

        self.input_dim = (self.num_state_var + self.num_action_var) * self.fc_dims[0]
        
        self.output_dim = self.lcm_dim_1 * self.lcm_dim_2
        if not self.learn_upper: 
            assert not self.learn_action
            self.output_dim = int(self.output_dim / 2)
        self.ones = torch.arange(self.num_state_var)
        self.lower_inds = np.tril_indices(self.num_state_var, -1)
        self.upper_inds = np.triu_indices(self.num_state_var, 1)

        self.code_dim = params.ours_params.code_dim
        self.codebook_size = params.ours_params.codebook_size

        if 'ours' in self.ours_type:
            enc_fc_dims = params.ours_params.vq_encode_fc_dims
            dec_fc_dims = params.ours_params.vq_decode_fc_dims
            
            encs = nn.Sequential()
            in_dim = self.input_dim
            for idx, out_dim in enumerate(enc_fc_dims):
                encs.add_module(f"fc_{idx}", nn.Linear(in_dim, out_dim, bias=False))
                encs.add_module(f"bn_{idx}", nn.BatchNorm1d(out_dim))
                encs.add_module(f"leaky_relu_{idx}", nn.LeakyReLU())
                in_dim = out_dim
            encs.add_module("fc_final", nn.Linear(in_dim, self.code_dim))
            self.encs = encs
            
            decs = nn.Sequential()
            in_dim = self.code_dim
            for idx, out_dim in enumerate(dec_fc_dims):
                decs.add_module(f"fc_{idx}", nn.Linear(in_dim, out_dim, bias=False))
                decs.add_module(f"bn_{idx}", nn.BatchNorm1d(out_dim))
                decs.add_module(f"leaky_relu_{idx}", nn.LeakyReLU())
                in_dim = out_dim
            decs.add_module("fc_final", nn.Linear(in_dim, self.output_dim))
            self.decs = decs
            
            self.apply(kaiming_init)

            self.ema = params.ours_params.vqvae_ema
            if self.ema:
                decay = params.ours_params.ema
                self.emb = NearestEmbedEMA(self.codebook_size, self.code_dim, decay=decay)
            else:
                self.emb = NearestEmbed(self.codebook_size, self.code_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reg_coef = params.ours_params.reg_coef
        self.vq_coef = params.ours_params.vq_coef
        self.commit_coef = params.ours_params.commit_coef
        
        self.code_index = []
        self.reset_loss()
        self.is_freeze = False

    def reset_loss(self):
        self.learned_local_mask = []
        self.reg_loss_list = []
        self.vq_loss_list = []
        self.commit_loss_list = []
        self.code_index = []
        self.reg_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        return self.encs(x)
    
    def adjust_dimension_default(self, z):
        bs = z.size(0)
        log_alpha = torch.zeros(bs, self.num_state_var, self.num_state_var + self.num_action_var, dtype=torch.float32, device=self.device)
        if self.learn_upper:
            z = z.reshape(bs, self.lcm_dim_1, self.lcm_dim_2)
            
            log_alpha[:, :, 1:1+self.lcm_dim_2] += torch.triu(z)
            log_alpha[:, :, :self.lcm_dim_2] += torch.tril(z, diagonal=-1)
            log_alpha[:, self.ones, self.ones] = 100
            
            if not self.learn_action:
                log_alpha[:, :, -self.num_action_var:] = 100
        else:
            log_alpha[:, self.lower_inds[0], self.lower_inds[1]] += z
            log_alpha[:, self.upper_inds[0], self.upper_inds[1]] = -100
            log_alpha[:, self.ones, self.ones] = 100
            log_alpha[:, :, -self.num_action_var:] = 100
        return log_alpha

    def decode(self, z):
        return self.adjust_dimension(self.decs(z))
    
    def get_codebook_local_mask(self):
        return torch.sigmoid(self.decode(self.emb.weight.t()))

    def total_loss(self):
        if self.is_freeze:
            return self.get_total_loss().detach()
        else:
            return self.get_total_loss()
    
    def get_total_loss(self):
        self.reg_loss = torch.stack(self.reg_loss_list, dim=-1).float()
        self.commit_loss = torch.stack(self.commit_loss_list, dim=-1).float()
        if self.ema: 
            self.vq_loss = torch.zeros_like(self.commit_loss)
        else:
            self.vq_loss = torch.stack(self.vq_loss_list, dim=-1).float()
        return self.reg_coef*self.reg_loss.mean() + self.vq_coef*self.vq_loss.mean() + self.commit_coef*self.commit_loss.mean()

    def loss_function(self, prob, z_e, emb):
        self.learned_local_mask.append(prob)
        self.reg_loss_list.append(prob.view(prob.size(0), -1).mean(dim=-1))
        
        if not self.ema: 
            self.vq_loss_list.append((emb - z_e.detach()).pow(2).mean())
        
        self.commit_loss_list.append((emb.detach() - z_e).pow(2).mean())
    
    def preprocess_ours_mask(self, feature, action):
        x = torch.cat([feature, action], dim=0)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        return x

    def forward_fcs(self, feature, action):
        x = self.preprocess(feature, action)
        z_e = self.encode(x)
        if self.ema:
            z_q, code_index = self.emb(z_e)
            emb = z_q.detach()
            z_q = z_e + (z_q - z_e).detach()
        else:
            z_q, code_index = self.emb(z_e, weight_sg=True)
            emb, _ = self.emb(z_e.detach())
        self.code_index.append(code_index)
        self.z_e = z_e
        return self.decode(z_q), z_e, emb

    def forward(self, feature, action, pred_step, tau=1, drawhard=True, training=True):
        if training:
            assert self.training
            assert self.emb.training
        else:
            assert not self.training
            assert not self.emb.training
        if self.is_freeze:
            assert self.training is False
            assert self.emb.training is False
        if pred_step == 0: 
            self.reset_loss()
        log_alpha, z_e, emb = self.forward_fcs(feature, action)
        prob = torch.sigmoid(log_alpha)
        if training and not self.is_freeze:
            sample = gumbel_sigmoid_batch(log_alpha, self.device, bs=self.local_mask_sampling_num, tau=tau, hard=drawhard)
        else:
            sample = (prob > 0.5).float().unsqueeze(0)
        
        self.loss_function(prob, z_e, emb)
        return sample, prob

class GumbelMatrix_NCD(VQVAEGumbelMatrixLatent):
    def __init__(self, params, feature_dim, action_dim, num_state_var, num_action_var, continuous_state, fc_dims, device):
        super(GumbelMatrix_NCD, self).__init__(params, feature_dim, action_dim, num_state_var, num_action_var, continuous_state, fc_dims, device)
        
        ncd_fc_dims = params.ours_params.ncd_fc_dims
        
        enc = []
        in_dim = self.input_dim
        for out_dim in ncd_fc_dims:
            enc.append(nn.Linear(in_dim, out_dim, bias=False))
            enc.append(nn.BatchNorm1d(out_dim))
            enc.append(nn.LeakyReLU())
            in_dim = out_dim
        enc.append(nn.Linear(in_dim, self.output_dim))
        self.encs = nn.Sequential(*enc)
        self.apply(kaiming_init)

    def total_loss(self):
        if self.is_freeze:
            return self.get_total_loss().detach()
        else:
            return self.get_total_loss()
    
    def get_total_loss(self):
        self.reg_loss = torch.stack(self.reg_loss_list, dim=-1).float()
        return self.reg_coef*self.reg_loss.mean()

    def loss_function(self, prob):
        self.learned_local_mask.append(prob)
        self.reg_loss_list.append(prob.view(prob.size(0), -1).mean(dim=-1))

    def encode(self, x):
        return self.encs(x)

    def forward_fcs(self, feature, action):
        x = self.preprocess(feature, action)
        z = self.encode(x)
        return self.adjust_dimension(z)

    def forward(self, feature, action, pred_step, tau=1, drawhard=True, training=True):
        if pred_step == 0: 
            self.reset_loss()
        log_alpha = self.forward_fcs(feature, action)
        prob = torch.sigmoid(log_alpha)
        assert training == self.training
        if training:
            sample = gumbel_sigmoid_batch(log_alpha, self.device, bs=self.local_mask_sampling_num, tau=tau, hard=drawhard)
        else:
            sample = (prob > 0.5).float().unsqueeze(0)
        self.loss_function(prob)
        return sample, prob