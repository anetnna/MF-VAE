import numpy as np
import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from jaxmarl.environments.spaces import Box, Discrete, MultiDiscrete

from typing import List, NamedTuple
from functools import partial

from .jax_buffer import JaxFbxBuffer


class Transition(NamedTuple):
    obs: dict
    next_obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict


def get_space_dim(space):
    # get the proper action/obs space from Discrete-MultiDiscrete-Box spaces
    if isinstance(space, (DiscreteGymnax, Discrete)):
        return space.n
    elif isinstance(space, (BoxGymnax, Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        print(space)
        raise NotImplementedError('Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces')


class JaxMARLWrapper(object):
    """
    Base class for all jaxmarl wrappers.
    Modified from JaxMARL official implementation of JaxMARLWrapper: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/baselines.py
    """

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


class EnvRolloutManager(JaxMARLWrapper):
    """ 
    Rollout Manageer for Training, suitable for both homogeneous and heterogeneous agents. Used by JaxMARL Q-learning.
    - Batchify multi environments (the number of parallel envs is defined by batch_size in __init__).
    - Add a golbal reward (rewrads["__all__"]) in the env.stp returns.
    - No padding observations of the agents.

    By default:
    - global_reward is the sum of all agents' rewards.

    Modified from JaxMARL official implementation of CTEnvRolloutManager: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/baselines.py
    """
    def __init__(self, env: MultiAgentEnv, batch_size: int, training_agents: List=None):
        super().__init__(env)
        self.batch_size = batch_size
        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = self.agents if training_agents is None else training_agents  

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            self.observation_spaces = {agent: self.observation_space() for agent in self.agents}
        if len(env.action_spaces) == 0:
            self.action_spaces = {agent: env.action_space() for agent in self.agents}
        
        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0)) for agent in self.agents}

        # valid actions
        self.valid_actions = {a:jnp.arange(u.n) for a, u in self.action_spaces.items()}

        # global rewards
        if 'smax' in env.name.lower() or 'overcooked' in env.name.lower():
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
        else:
            self.global_reward = lambda rewards: jnp.stack([rewards[agent] for agent in self.training_agents]).sum(axis=0)

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)
    
    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        return self._env.reset(key)

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        if 'hanabi' in self._env.name.lower():
            actions = jax.tree_util.tree_map(lambda x:jnp.expand_dims(x, 0), actions)
        obs, state, reward, done, infos = self._env.step(key, state, actions)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos
    
    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size)).astype(int)

    def get_action_space_dim(self):
        return {agent_id: get_space_dim(env.action_space(agent_id)) for agent_id in self.agents}



if __name__ == "__main__":
    # make parallel environment
    key = jax.random.PRNGKey(0)
    env = make('MPE_simple_tag_v3',
                num_good_agents=10,
                num_adversaries=30,
                num_obs=20,)
    wrapped_env = EnvRolloutManager(env, batch_size=8)

    # reset parallel env
    obs, state = wrapped_env.batch_reset(key)
    print(obs['adversary_0'].shape) # obs shape: (batch_size, obs_dim)

    # sample actions for parallel env
    key_act = jax.random.split(key, env.num_agents)
    actions = {agent: wrapped_env.batch_sample(key_act[i], agent) \
        for i, agent in enumerate(env.agents)}
    
    # step forward in parallel env
    key_step = jax.random.PRNGKey(42)
    next_obs, state, reward, done, infos = wrapped_env.batch_step(key_step, state, actions)
    print(obs['adversary_0'].shape) # obs shape: (batch_size, obs_dim)
    print(reward['__all__'])
    print(done['__all__'])

    # test interaction with buffer
    buffer = JaxFbxBuffer(max_length=10_000, 
                          min_length=64, 
                          batch_size=256, 
                          add_batch=True)
    obs_unbatched = jax.tree_map(lambda x: x[0, :], obs)
    reward_unbatched = jax.tree_map(lambda x: x[0, ], reward)
    actions_unbatched = jax.tree_map(lambda x: x[0], actions)
    next_obs_unbatched = jax.tree_map(lambda x: x[0, :], next_obs)
    done_unbatched = jax.tree_map(lambda x: x[0], done)
    buffer.init_buffer(obs_unbatched, reward_unbatched, actions_unbatched, next_obs_unbatched, done_unbatched)
    # buffer.buffer.init(sample_transitions_unbatched)

    for i in range(256 // 8):
        obs = next_obs
        actions = {agent: wrapped_env.batch_sample(key_act[i], agent) \
            for i, agent in enumerate(env.agents)}
        next_obs, state, reward, done, infos = wrapped_env.batch_step(key_step, state, actions)
        buffer.add_trans(obs, reward, actions, next_obs, done, batch_input=True)
    print(buffer.can_sample())

    key_sample = jax.random.PRNGKey(42)
    transitions = buffer.sample(key_sample)
    print(transitions.experience['agent_0_obs'].shape)

