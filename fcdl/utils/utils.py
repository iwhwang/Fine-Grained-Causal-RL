import functools
import json
import os
import random
import sys
import time
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..env.chemical_env import Chemical
from .multiprocessing_env import SubprocVecEnv


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self._clean_dict(self)
    
    def __deepcopy__(self, memo):
        return AttrDict(deepcopy(dict(self), memo))

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if isinstance(v, str) and v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = AttrDict(self._clean_dict(v))
            if isinstance(v, list):
                v = [AttrDict(self._clean_dict(_v)) if isinstance(_v, dict) else _v for _v in v]
            _dict[k] = v
        return _dict


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


class TrainingParams(AttrDict):
    def __init__(self, training_params_fname="params.json", train=True):
        config = json.load(open(training_params_fname))
        super(TrainingParams, self).__init__(config)

        repo_path = os.path.dirname(self.__dict__["wandb_dir"])
        training_params = self.training_params

        if train:
            if training_params_fname == "policy_params.json":
                sub_dirname = "task" if training_params.rl_algo == "model_based" else "dynamics"
            else:
                raise NotImplementedError

            info = self.info.replace(" ", "_")
            experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")

            self.replay_buffer_dir = None
            if training_params_fname == "policy_params.json" and training_params.replay_buffer_params.saving_freq:
                self.replay_buffer_dir = os.path.join(repo_path, "replay_buffer", experiment_dirname)
                os.makedirs(self.replay_buffer_dir)




def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def override_params_from_cli_args(params):
    """
    e.g.) `python main_policy.py --cuda_id=2 --env_params.chemical_env_params.num_objects=10`
    """
    args = sys.argv

    if len(args) > 1:
        for arg in args[1:]:
            keys, v = arg.split("=")
            keys = keys.split("--")[1].split(".")
            param = params
            for k in keys[:-1]:
                param = getattr(param, k)
            param_v_type = type(getattr(param, keys[-1]))
            if param_v_type is type(bool) or v.capitalize() in ["False", "True"]:
                if v.capitalize() == "False":
                    setattr(param, keys[-1], False)
                elif v.capitalize() == "True":
                    setattr(param, keys[-1], True)
                else:
                    raise ValueError("Invalid value")
            else:
                if param_v_type is type(None):
                    param_v_type = str
                setattr(param, keys[-1], param_v_type(v))


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def preprocess_obs(obs, params):
    """
    filter unused obs keys, convert to np.float32 / np.uint8, resize images if applicable
    """
    def to_type(ndarray, type):
        if ndarray.dtype != type:
            ndarray = ndarray.astype(type)
        return ndarray

    obs_spec = getattr(params, "obs_spec", obs)
    new_obs = {}
    
    for k in params.obs_keys + params.goal_keys:
        val = obs[k]
        val_spec = obs_spec[k]
        if val_spec.ndim == 1:
            val = to_type(val, np.float32)
        if val_spec.ndim == 3:
            num_channel = val.shape[2]
            if num_channel == 1:
                env_params = params.env_params
                assert "Causal" in env_params.env_name
                val = to_type(val, np.float32)
            elif num_channel == 3:
                val = to_type(val, np.uint8)
            else:
                raise NotImplementedError
            val = val.transpose((2, 0, 1))                  # c, h, w
        new_obs[k] = val
    return new_obs


def postprocess_obs(obs):
    # convert images to float32 and normalize to [0, 1]
    new_obs = {}
    for k, val in obs.items():
        if val.dtype == np.uint8:
            val = val.astype(np.float32) / 255
        new_obs[k] = val
    return new_obs


def hamming_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return (x != y).float().sum(-1).mean().item()


def update_obs_act_spec(env, params):
    """
    get act_dim and obs_spec from env and add to params
    """
    params.continuous_state = params.continuous_action = params.continuous_factor = not params.env_params.env_name in ["Physical", "Chemical"]
    if params.encoder_params.encoder_type == "conv":
        params.continuous_state = True
    
    params.action_dim = env.action_dim
    # params.action_inner_dim = env.action_inner_dim
    params.feature_inner_dim = env.feature_inner_dim
    params.obs_spec = obs_spec = preprocess_obs(env.observation_spec(), params)
    params.num_action_variable = env.num_action_variable
        
    if params.continuous_factor:
        params.obs_dims = None
        params.action_spec = env.action_spec
    else:
        params.obs_dims = obs_dims = env.observation_dims()
        params.action_spec = None


def get_single_env(params, load_dir=None, test_idx=None, env_idx=None):
    env_params = params.env_params
    env_name = env_params.env_name
    if env_idx is not None:
        if params.seed == -1:
            seed = (os.getpid() * int(time.time())) % 123456789
        else:
            seed = params.seed + env_idx * 10000
        set_seed_everywhere(seed)
    if env_name == "Chemical":
        copied_params = deepcopy(params)
        copied_env_params = copied_params.env_params.chemical_env_params
        if test_idx is not None:
            test_params = copied_env_params.test_params[test_idx]
            for k, v in test_params.items():
                setattr(copied_env_params, k, v)
        if env_idx is not None:
            copied_env_params.name += f"_{str(env_idx)}"
        env = Chemical(copied_params, load_dir)
    else:
        raise ValueError("Unknown env_name: {}".format(env_name))

    return env


def get_subproc_env(params, load_dir, test_idx=None, env_idx=None):
    def get_single_env_wrapper():
        return get_single_env(params, load_dir, test_idx, env_idx)
    return get_single_env_wrapper


def get_env(params, load_dir=None, test_idx=None):
    num_env = params.env_params.num_env
    env_name = params.env_params.env_name
    if num_env == 1:
        return get_single_env(params, load_dir, test_idx)
    else:
        assert "Chemical" == env_name
        return SubprocVecEnv([get_subproc_env(params, load_dir, test_idx, env_idx) for env_idx in range(num_env)])


def get_start_step_from_model_loading(params):
    """
    if model-based policy is loaded, return its training step;
    elif inference is loaded, return its training step;
    else, return 0
    """
    task_learning = params.training_params.rl_algo == "model_based"
    load_model_based = params.training_params.load_model_based
    load_inference = params.training_params.load_inference
    if load_model_based is not None and os.path.exists(load_model_based):
        model_name = load_model_based.split(os.sep)[-1]
        # start_step = int(model_name.split("_")[-1])
        start_step_k = model_name.split("_")[-1][:-1] # ex) 100k
        start_step = int(start_step_k) * 1000
        print("start_step:", start_step)
    elif load_inference is not None and os.path.exists(load_inference) and not task_learning:
        model_name = load_inference.split(os.sep)[-1]
        # start_step = int(model_name.split("_")[-1])
        start_step_k = model_name.split("_")[-1][:-1] # ex) 100k
        start_step = int(start_step_k) * 1000
        print("start_step:", start_step)
    else:
        start_step = 0
    return start_step
