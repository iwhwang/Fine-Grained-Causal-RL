import os
import time
import torch
import numpy as np

from ..utils.utils import preprocess_obs, postprocess_obs, rgetattr


def take(array, start, end):
    """
    get array[start:end] in a circular fashion
    """
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    return array[idxes]


def assign(array, start, end, value):
    if start >= end:
        end += len(array)
    idxes = np.arange(start, end) % len(array)
    array[idxes] = value


class ReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.continuous_action = params.continuous_action
        self.continuous_factor = params.continuous_factor

        training_params = params.training_params
        replay_buffer_params = training_params.replay_buffer_params
        self.capacity = capacity = replay_buffer_params.capacity
        # each transition can be sampled at most max_sample_time times for inference and policy training
        self.max_sample_time = replay_buffer_params.max_sample_time
        self.saving_freq = replay_buffer_params.saving_freq

        self.saving_dir = params.replay_buffer_dir
        self.n_inference_pred_step = params.inference_params.n_pred_step

        obs_spec = params.obs_spec
        action_dim = params.action_dim
        self.obses = {k: np.empty((capacity, *v.shape), dtype=v.dtype) for k, v in obs_spec.items()}
        if self.continuous_action:
            self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.is_trains = np.empty((capacity, 1), dtype=bool)

        if params.env_params.env_name == "Chemical":
            self.root_color = np.empty((capacity), dtype=np.int64)
            if rgetattr(self.params, 'env_params.chemical_env_params.gt_local_mask', False):
                num_objects = params.env_params.chemical_env_params.num_objects
                self.lcms = np.empty((capacity, num_objects, num_objects + 1), dtype=np.float32)

        self.inference_sample_times = np.zeros(capacity)
        self.policy_sample_times = np.zeros(capacity)
        self.model_based_sample_times = np.zeros(capacity)

        self.idx = 0
        self.last_save = 0
        self.full = False

        load_replay_buffer = training_params.load_replay_buffer
        if load_replay_buffer is not None and os.path.isdir(load_replay_buffer):
            self.load(load_replay_buffer)

        self.num_env = params.env_params.num_env
        self.is_vecenv = self.num_env > 1
        if self.is_vecenv:
            self.temp_buffer = [[] for _ in range(self.num_env)]

    def add(self, obs, action, reward, next_obs, done, info, is_train):
        if self.is_vecenv:
            for i in range(self.num_env):
                obs_i = {key: val[i] for key, val in obs.items()}
                self.temp_buffer[i].append([obs_i, action[i], reward[i], done[i], info[i], is_train[i]])
                if done[i]:
                    for obs_, action_, reward_, done_, info_, is_train_ in self.temp_buffer[i]:
                        self._add(obs_, action_, reward_, done_, info_, is_train_)
                    final_obs = info[i]["obs"]
                    # use done = -1 as a special indicator that the added obs is the last obs in the episode
                    self._add(final_obs, action_, 0, -1, info_, is_train_)
                    self.temp_buffer[i] = []
        else:
            self._add(obs, action, reward, done, info, is_train)
            if done:
                # use done = -1 as a special indicator that the added obs is the last obs in the episode
                self._add(next_obs, action, 0, -1, info, is_train)

    def _add(self, obs, action, reward, done, info, is_train):
        raw_obs = obs
        obs = preprocess_obs(obs, self.params)
        for k in obs.keys():
            np.copyto(self.obses[k][self.idx], obs[k])

        if self.params.env_params.env_name == "Chemical":
            self.root_color[self.idx] = int(obs['obj0'][0])
            if rgetattr(self.params, 'env_params.chemical_env_params.gt_local_mask', False):
                np.copyto(self.lcms[self.idx], info["lcm"])

        if self.continuous_action and action.dtype != np.float32:
            action = action.astype(np.float32)
        elif not self.continuous_action:
            action = np.int64(action)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.is_trains[self.idx], is_train)

        self.inference_sample_times[self.idx] = 0
        self.policy_sample_times[self.idx] = 0
        self.model_based_sample_times[self.idx] = 0

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

        # if (self.saving_freq > 0) and (self.idx % self.saving_freq == 0):
        #     self.save(self.saving_dir)

    def valid_idx(self, idx, n_step, type, use_part="all"):
        if use_part != "all":
            is_train = self.is_trains[idx]
            if use_part == "train" and not is_train:
                return False
            if use_part == "eval" and is_train:
                return False

        is_local_sample = True
        if type == "policy":
            if self.policy_sample_times[idx] >= self.max_sample_time:
                return False
        elif type == "inference":
            if self.inference_sample_times[idx] >= self.max_sample_time:
                return False
        elif type == "ood":
            if self.params.env_params.env_name == "Chemical":
                is_local_sample = (take(self.root_color, idx, idx + n_step + 1) == 0).all()
            else:
                raise NotImplementedError
        else:
            if self.model_based_sample_times[idx] >= self.max_sample_time:
                return False

        not_at_episode_end = (take(self.dones, idx, idx + n_step) != -1).all()
        not_newly_added = (idx >= self.idx) or ((idx + n_step) % self.capacity < self.idx)
        return not_at_episode_end and not_newly_added and is_local_sample

    def sample_idx(self, batch_size, n_step, type, use_part="all"):
        idxes = []
        sample_start_time = time.time()
        for _ in range(batch_size):
            while True:
                idx = np.random.randint(self.capacity if self.full else (self.idx - n_step))
                if self.valid_idx(idx, n_step, type, use_part):
                    idxes.append(idx)

                    if type == "inference" and use_part != "eval":
                        self.inference_sample_times[idx] += 1
                    elif type == "policy":
                        self.policy_sample_times[idx] += 1
                    elif type == "ood":
                        pass
                    else:
                        self.model_based_sample_times[idx] += 1

                    break
                
                cur_time = time.time()
                # If we have been sampling for more than 1 minute, raise an error
                if cur_time - sample_start_time > 60:
                    raise RuntimeError("Unable to sample a valid transition after 1 minute. ")
        return np.array(idxes)

    def construct_transition(self, idxes, n_step, type):
        obses = postprocess_obs({k: v[idxes] for k, v in self.obses.items()})
        obses = {k: torch.from_numpy(v).to(self.device) for k, v in obses.items()}

        actions = torch.tensor(np.array([take(self.actions, idx, idx + n_step) for idx in idxes]),
                               dtype=torch.float32 if self.continuous_action else torch.int64, device=self.device)
        rewards = torch.tensor(np.array([take(self.rewards, idx, idx + n_step) for idx in idxes]),
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[(idxes + n_step - 1) % self.capacity],
                             dtype=torch.float32, device=self.device)
        info = dict()

        if self.params.env_params.env_name in ["Chemical"]:
            lcms = torch.tensor(np.array([take(self.lcms, idx, idx + n_step) for idx in idxes]),
                                    dtype=torch.float32, device=self.device)
            info["lcms"] = lcms

        next_obses = None
        # if type != "model_based":
        if True:
            next_obses = postprocess_obs({k: np.array([take(v, idx + 1, idx + n_step + 1) for idx in idxes])
                                          for k, v in self.obses.items()})
            next_obses = {k: torch.tensor(v, device=self.device) for k, v in next_obses.items()}

        return obses, actions, rewards, dones, next_obses, info

    def sample(self, batch_size, type, use_part="all"):
        """
        Sample training data for inference model
        return: obses: (bs, obs_spec)
        return: actions: (bs, n_step, action_dim)
        return: next_obses: (bs, n_step, obs_spec)
        """
        assert type in ["inference", "model_based", "encoder", "ood"], "Unrecognized sample type: {}".format(type)

        if type == "inference":
            n_step = self.n_inference_pred_step
        elif type in ["model_based", "encoder", "ood"]:
            n_step = 1
        else:
            raise NotImplementedError

        idxes = self.sample_idx(batch_size, n_step, type, use_part)
        obses, actions, rewards, dones, next_obses, info = self.construct_transition(idxes, n_step, type)
        return obses, actions, rewards, dones, next_obses, idxes, info

    def sample_ood_eval(self, batch_size, use_part="all"):
        obses, actions, _, _, next_obses, _, info = self.sample(batch_size, "ood", use_part)
        return obses, actions, next_obses, info

    def sample_inference(self, batch_size, use_part="all"):
        obses, actions, _, _, next_obses, _, info = self.sample(batch_size, "inference", use_part)
        return obses, actions, next_obses, info

    def sample_policy(self, batch_size):
        raise NotImplementedError


    def save(self, save_dir, easy_step):
        env_name = self.params.env_params.env_name
        if self.idx == self.last_save:
            return
        # path = os.path.join(save_dir, "%d_%d.p" % (self.last_save, self.idx))
        path = os.path.join(save_dir, f"replay_buffer_{easy_step}k")
        payload = {"obses": {k: take(v, self.last_save, self.idx) for k, v in self.obses.items()},
                   "actions": take(self.actions, self.last_save, self.idx),
                   "rewards": take(self.rewards, self.last_save, self.idx),
                   "dones": take(self.dones, self.last_save, self.idx),
                   "is_trains": take(self.is_trains, self.last_save, self.idx),
                   "inference_sample_times": take(self.inference_sample_times, self.last_save, self.idx),
                   "policy_sample_times": take(self.policy_sample_times, self.last_save, self.idx),
                   "model_based_sample_times": take(self.model_based_sample_times, self.last_save, self.idx)}
        if (env_name == "Chemical" and rgetattr(self.params, 'env_params.chemical_env_params.gt_local_mask', False)):
            payload["lcms"] = take(self.lcms, self.last_save, self.idx)
        
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = [os.path.join(save_dir, chunk) for chunk in os.listdir(save_dir)]
        chunks.sort(key=os.path.getctime)
        env_name = self.params.env_params.env_name
        for chunk in chunks:
            chunk_fname = os.path.split(chunk)[1]
            start, end = [int(x) for x in chunk_fname.split(".")[0].split("_")]
            payload = torch.load(chunk)
            for k, v in payload["obses"].items():
                assign(self.obses[k], start, end, v)
            assign(self.actions, start, end, payload["actions"])
            assign(self.rewards, start, end, payload["rewards"])
            assign(self.dones, start, end, payload["dones"])
            assign(self.is_trains, start, end, payload["is_trains"])
            if (env_name == "Chemical" and rgetattr(self.params, 'env_params.chemical_env_params.gt_local_mask', False)):
                assign(self.lcms, start, end, payload["lcms"])
            
            assign(self.inference_sample_times, start, end, payload["inference_sample_times"])
            assign(self.policy_sample_times, start, end, payload["policy_sample_times"])
            assign(self.model_based_sample_times, start, end, payload["model_based_sample_times"])
            self.idx = end
            if end < start:
                self.full = True
        if len(chunks):
            # episode ends
            self.dones[self.idx - 1] = -1
        print("replay buffer loaded from", save_dir)

    def __len__(self):
        return self.capacity if self.full else (self.idx + 1)