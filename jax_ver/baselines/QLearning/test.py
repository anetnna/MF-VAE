"""
Test between different algorithms
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from functools import partial
from typing import NamedTuple, Dict, Union, Any
from collections import defaultdict

import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_dir)

from src.env import get_space_dim, EnvRolloutManager
from jaxmarl import make
from src.jax_buffer import JaxFbxTrajBuffer, Transition
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import chex
import optax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf
import json

from utils.RNN import ScannedRNN, EpsilonGreedy, AgentRNN
from utils.load_save_model import load_params, reconstruct_params


def get_homogeneous_group(agents_id):
    # categories = set(agent_id.split('_')[0] for agent_id in agents_id)
    category_counts = {}
    for agent_id in agents_id:
        agent_category = agent_id.split('_')[0]
        if agent_category not in category_counts:
            category_counts[agent_category] = 0
        category_counts[agent_category] += 1
    return category_counts


def make_test(config, agent_net_param, env):
    def test(rngs):
        agent_groups_id_count = get_homogeneous_group(env.agents)
        rng, _rng = jax.random.split(jax.random.PRNGKey(42))
        test_env = EnvRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"])
        agent_net = {}
        act_dim = {}
        for agent_group_id in agent_groups_id_count.keys():
            act_dim[agent_group_id] = get_space_dim(test_env.action_space(agent_group_id+"_0"))
            agent_net[agent_group_id] = AgentRNN(action_dim=act_dim[agent_group_id], 
                                                hidden_dim=config["AGENT_HIDDEN_DIM"], 
                                                init_scale=config['AGENT_INIT_SCALE'])
        
        def homogeneous_group_pass(params: Any, hidden_state: Any,
                                   obs: Dict[str, Any],
                                   dones: Dict[str, Any],
                                   group_id: str):
            agents = [key for key in obs.keys() if key.startswith(group_id)]
            flatten_agents_obs = [obs[key] for key in agents]
            original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
            batched_input = (
                jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
            )
            hidden_state, q_vals = agent_net[group_id].apply(params, hidden_state, batched_input)
            q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
            q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
            return hidden_state, q_vals
        
        def get_greedy_metrics(test_state, unused):
            rng = test_state[0]
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                actions_all = {}
                for agent_group_id in agent_groups_id_count.keys():
                    obs_   = {a:last_obs[a] for a in env.agents if a.startswith(agent_group_id)}
                    obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                    dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                    hstate[agent_group_id], q_vals = homogeneous_group_pass(params[agent_group_id], 
                                                                            hstate[agent_group_id], 
                                                                            obs_, 
                                                                            dones_, 
                                                                            agent_group_id)
                    valid_actions = {k:v for k, v in test_env.valid_actions.items() if k.startswith(agent_group_id)}
                    actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, valid_actions)
                    actions_all.update(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions_all)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)
            
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            hstate = {}
            for agent_group_id in agent_groups_id_count.keys():
                hstate_tmp = ScannedRNN.initialize_carry(
                    config['AGENT_HIDDEN_DIM'], 
                    agent_groups_id_count[agent_group_id]*config["NUM_TEST_EPISODES"]
                ) # (n_agents*n_envs, hs_size)
                hstate[agent_group_id] = hstate_tmp

            step_state = (
                agent_net_param,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            _, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )

            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = dones['__all__']
            first_returns = jax.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
            first_infos   = jax.tree_map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
            metrics = {
                **{'test_returns_'+k:v.mean() for k, v in first_returns.items()},
                **{'test_'+k:v for k,v in first_infos.items()}
            }

            return (rng, ), metrics
        
        # time_state = {'timesteps':jnp.array(0),
        #     'updates':  jnp.array(0)
        # }
        test_state = (_rng, )
        _, test_metrics = get_greedy_metrics(test_state, None) # initial greedy metrics
        # print(test_metrics)

        _, metrics = jax.lax.scan(
            get_greedy_metrics, test_state, None, config["TEST_TRUNS"]
        )
        return metrics
    
    return test


if __name__ == "__main__":
    config = {}
    config["NUM_TEST_EPISODES"] = 32
    config['AGENT_HIDDEN_DIM'] = 64
    config["NUM_STEPS"] = 25
    config["TEST_TRUNS"] = 128
    config["NUM_SEEDS"] = 2
    config['AGENT_INIT_SCALE'] = 2.
    env_name = "MPE_simple_tag_v3"
    env = make(env_name,
               num_good_agents=10,
               num_adversaries=30,
               num_obs=20,)
    # env = EnvRolloutManager(plain_env, batch_size=env_batch_size)
    agents_id = env.agents
    agent_id_codebook = {}
    for i, agent_id in enumerate(agents_id):
        agent_id_codebook[agent_id] = i

    save_path = "/home/huaze/enze/jax-mf-vae/jax_ver/baselines/QLearning/checkpoints/"
    # model_role = {"qmix": "agent", "iql": "adversary"}
    model_role = {"agent": "iql",  "adversary": "qmix"}
    # model_select = list(model_role.keys())
    # params_keys = list(model_role.values())
    model_select = list(set(model_role.values()))
    params_keys = list(model_role.keys())
    # model_loaded = {}
    for i, model_name in enumerate(model_select):
        save_dir = os.path.join(save_path, env_name, model_name)
        model = {}
        for k in params_keys:
            flattened_params = load_params(f'{save_dir}/{k}.safetensors')
            params = reconstruct_params(flattened_params)
            if "qmix" in model_name:
                model[k] = params["agentRNN"]
            else:
                model[k] = params
            # model_loaded[model_select[i]] = model
            # model_loaded[k] = model
    # merged_model = {k: model_loaded[i][k] for i, k in enumerate(model_role)}
    merged_model = model
    # for k, v in model_role.items():
    #     merged_model[v] = model_loaded[k][v]
    print(merged_model.keys())

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # test_func = make_test(config, merged_model, env)
    # outs = test_func(rngs)
    test_vjit = jax.jit(jax.vmap(make_test(config, merged_model, env)))
    outs = jax.block_until_ready(test_vjit(rngs))

    # print(outs)
    cut_name_func = lambda s: '_'.join(s.split('_')[2:])
    output_for_save = {cut_name_func(k): np.array(v).reshape(-1) for k, v in outs.items() if "all" not in k}
    data = list(output_for_save.values())
    labels = list(output_for_save.keys())

    save_file = save_path
    for k, v in model_role.items():
        save_file += f"{k}_{v}_"
    save_file += "results.jpg"

    fig_title = ""
    for k, v in model_role.items():
        fig_title += f"{k}_{v} "
    
    plt.boxplot(data, labels=labels)
    plt.title(fig_title)
    plt.ylabel('Rewards')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    plt.savefig(save_file)

    # with open(save_file, 'w') as f:
    #     json.dump(outs, f)


