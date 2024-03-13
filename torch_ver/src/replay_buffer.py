from cpprb import ReplayBuffer
from functools import partial
import jax
import gymnasium
import numpy as np

import flashbax as fbx

class JaxUniformReplayBuffer():
    def __init__(
        self,
        buffer_state: dict,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_state = buffer_state
    
    @partial(jax.jit, static_argnums=(0))
    def add(
        self,
        experience: tuple,
        idx: int,
    ):
        state, action,reward, next_sate, done = experience
        idx = idx % self.buffer_size
        self.buffer_state["state"] = self.buffer_state["state"].at[idx].set(state)
        self.buffer_state["action"] = self.buffer_state["action"].at[idx].set(action)
        self.buffer_state["reward"] = self.buffer_state["reward"].at[idx].set(reward)
        self.buffer_state["next_sate"] = self.buffer_state["next_sate"].at[idx].set(next_sate)
        self.buffer_state["done"] = self.buffer_state["done"].at[idx].set(done)


def get_space_shape(space_item):
    if isinstance(space_item, gymnasium.spaces.Discrete):
        # space_shape = space_item.n
        return (1,)
    elif isinstance(space_item, gymnasium.spaces.MultiBinary):
        space_shape = space_item.n
        if isinstance(space_shape, int):
            return (space_shape,)
        elif isinstance(space_shape, tuple):
            return space_shape
        else:
            raise NotImplementedError
    elif isinstance(space_item, gymnasium.spaces.Box):
        return space_item.shape
    else:
        raise NotImplementedError


class MultiAgentCPPRB:
    def __init__(self, environment, max_size=10000, batch_size=32):
        self._environment = environment
        self._max_size = max_size
        self._batch_size = batch_size

        cpprb_env_dict = {}
        buffer = {}

        for agent_id in environment.agents:
            obs_shape = get_space_shape(self._environment.observation_space(agent_id))
            act_shape = get_space_shape(self._environment.action_space(agent_id))

            cpprb_env_dict[f"{agent_id}_observations"] = {"shape": obs_shape}
            cpprb_env_dict[f"{agent_id}_next_observations"] = {"shape": obs_shape}
            cpprb_env_dict[f"{agent_id}_actions"] = {"shape": act_shape}
            cpprb_env_dict[f"{agent_id}_rewards"] = {"shape": (1,)}
            cpprb_env_dict[f"{agent_id}_terminals"] = {"shape": (1,)}
            cpprb_env_dict[f"{agent_id}_truncations"] = {"shape": (1,)}
        
            buffer[f"{agent_id}_observations"] = np.zeros(obs_shape, "float32")
            buffer[f"{agent_id}_next_observations"] = np.zeros(obs_shape, "float32")
            buffer[f"{agent_id}_actions"] = np.zeros(act_shape, "float32")
            buffer[f"{agent_id}_rewards"] = np.zeros((1,), "float32")
            buffer[f"{agent_id}_terminals"] = np.zeros((1,), "float32")
            buffer[f"{agent_id}_truncations"] = np.zeros((1,), "float32")

        cpprb_env_dict["mask"] = {"shape": (1,)}
        buffer["mask"] = np.zeros((1,), "float32")

        self._cpprb = ReplayBuffer(max_size, env_dict=cpprb_env_dict, default_type=np.float32)
        self._buffer = buffer
        # self._t = 0
    
    def add(self, observations, next_observations, actions, rewards, terminals, trunctions):
        for agent_id in self._environment.agents:
            # self._buffer[f"{agent_id}_observations"][self._f] = np.array(observations[agent_id], "float32")
            # self._buffer[f"{agent_id}_actions"][self._f] = np.array(actions[agent_id], "float32")
            # self._buffer[f"{agent_id}_rewards"][self._f] = np.array(rewards[agent_id], "float32")
            # self._buffer[f"{agent_id}_terminals"][self._f] = np.array(terminals[agent_id], "float32")
            # self._buffer[f"{agent_id}_truncations"][self._f] = np.array(trunctions[agent_id], "float32")
            self._buffer[f"{agent_id}_observations"] = np.array(observations[agent_id], "float32")
            self._buffer[f"{agent_id}_next_observations"] = np.array(next_observations[agent_id], "float32")
            self._buffer[f"{agent_id}_actions"] = np.array(actions[agent_id], "float32")
            self._buffer[f"{agent_id}_rewards"] = np.array(rewards[agent_id], "float32")
            self._buffer[f"{agent_id}_terminals"] = np.array(terminals[agent_id], "float32")
            self._buffer[f"{agent_id}_truncations"] = np.array(trunctions[agent_id], "float32")
        
        # self._t += 1
        self._cpprb.add(**self._buffer)
    
    def on_episode_end(self):
        self._cpprb.on_episode_end()
    
    def sample(self):
        return self._cpprb.sample(self._batch_size)

    def __iter__(self):
        return self
    
    def __next__(self):
        cpprb_sample = self._cpprb.sample(self._batch_size)
        return cpprb_sample


