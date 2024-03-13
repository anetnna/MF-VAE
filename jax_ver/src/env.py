import numpy as np
from pettingzoo.mpe import simple_tag_v3
import gymnasium
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State


def get_space_size(space_item):
    if isinstance(space_item, gymnasium.spaces.Discrete):
        space_shape = space_item.n
        return (space_shape,)
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


def create_env(env_name):
    # create environment
    if env_name == 'simple_tag_v3':
        parallel_env = simple_tag_v3.parallel_env(num_good=10, num_adversaries=30, num_obstacles=20, max_cycles=1000, continuous_actions=False)
    else:
        raise NotImplementedError
    obs, infos = parallel_env.reset(seed=42)
    print(f"Info about environment: agents in env: {parallel_env.agents}")
    obs_dim_all = {}
    act_dim_all = {}
    for agent_id in parallel_env.agents:
        print(f"Agent {agent_id} info: \t action space size: {parallel_env.action_space(agent_id)}, observation space size: {parallel_env.observation_space(agent_id)}")
        obs_dim_all[agent_id] = get_space_size(parallel_env.observation_space(agent_id))[0]
        act_dim_all[agent_id] = get_space_size(parallel_env.action_space(agent_id))[0]
    
    return parallel_env, obs_dim_all, act_dim_all, obs, infos


def create_transition(obs, action, next_obs, done, rew):
    obs_all = []
    action_all = []
    next_obs_all = []
    done_all = []
    for obs_sa in obs.keys():
        obs_all.append(obs[obs_sa])
        action_all.append(action[obs_sa])
        next_obs_all.append(next_obs[obs_sa])
        done_all = True if True in done.values() else False
    
    obs_all = np.array(obs_all)
    action_all = np.array(action_all)
    next_obs_all = np.array(next_obs_all)

    return obs_all, action_all, next_obs_all, done_all, rew


class EnvRolloutManager:
    def __init__(self):
        pass

