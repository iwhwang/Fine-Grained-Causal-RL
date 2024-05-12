# Code is from OpenAI baseline: https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import logging
import numpy as np
from multiprocessing import Process, Pipe, shared_memory


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                info["obs"] = ob
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'observation_spec':
            remote.send(env.observation_spec())
        elif cmd == 'workspace_spec':
            remote.send(env.workspace_spec())
        elif cmd == 'save_mlps':
            env.save_mlps()
            remote.send(True)
        elif cmd == 'load_mlps':
            env.load_mlps()
            remote.send(True)
        elif cmd.startswith("get_attr_"):
            attr_name = cmd[len("get_attr_"):]
            attr = getattr(env, attr_name, None)
            if attr is None:
                logging.warning("Attribute {} not found in env".format(attr_name))
            remote.send(attr)
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True     # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        VecEnv.__init__(self, len(env_fns))

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs = {key: np.stack([d[key] for d in obs]) for key in obs[0].keys()}
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = {key: np.stack([d[key] for d in obs]) for key in obs[0].keys()}
        return obs

    def reset_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def observation_spec(self):
        self._assert_not_closed()
        self.remotes[0].send(('observation_spec', None))
        return self.remotes[0].recv()

    def workspace_spec(self):
        self._assert_not_closed()
        self.remotes[0].send(('workspace_spec', None))
        return self.remotes[0].recv()

    def __getattr__(self, name):
        self._assert_not_closed()
        if 'magnetic_force' in name:
            for remote in self.remotes:
                remote.send((f'get_attr_{name}', None))
            res = [remote.recv() for remote in self.remotes]
        else: 
            self.remotes[0].send(('get_attr_{}'.format(name), None))
            res = self.remotes[0].recv()
        return res

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def close(self):
        if self.closed:
            return
        self.close_extras()

    def __len__(self):
        return self.nenvs
