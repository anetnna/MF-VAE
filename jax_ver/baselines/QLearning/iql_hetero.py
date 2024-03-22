"""
End-to-End JAX Implementation of IQL.

Notice:
- Agents are controlled by a single RNN architecture.
- Sharing parameters among the same agents.
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can use TD Loss (pymarl2) or DDQN loss (pymarl)
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
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
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from utils.RNN import ScannedRNN, EpsilonGreedy, AgentRNN


def get_homogeneous_group(agents_id):
    # categories = set(agent_id.split('_')[0] for agent_id in agents_id)
    category_counts = {}
    for agent_id in agents_id:
        agent_category = agent_id.split('_')[0]
        if agent_category not in category_counts:
            category_counts[agent_category] = 0
        category_counts[agent_category] += 1
    return category_counts


def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    def train(rngs):
        # INIT ENV
        # rng, _rng = jax.random.split(rngs)
        rng, _rng = jax.random.split(jax.random.PRNGKey(42))
        wrapped_env = EnvRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = EnvRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"])
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        # remove the NUM_ENV dim, after size: [traj_len, ...]
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) 
        buffer = JaxFbxTrajBuffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'], 
            min_length_time_axis=config['BUFFER_BATCH_SIZE'], 
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config["NUM_ENVS"], # for samplified, add_batch_size=
            sample_sequence_length=1,
            period=1,
        )
        buffer.init_buffer(sample_traj_unbatched.obs, 
                           sample_traj_unbatched.rewards, 
                           sample_traj_unbatched.actions, 
                           sample_traj_unbatched.dones)
        
        # INIT NETWORK
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=config['EPS_ADAM'])
        )

        agent_groups_id_count = get_homogeneous_group(env.agents)
        # agent_groups_id_count be like: {'adversity': 20, 'agent': 10}
        rng, _rng = jax.random.split(rng)
        agent_net = {}
        obs_dim = {}
        act_dim = {}
        network_params = {}
        train_state_all = {}
        target_agent_params = {} # target network params
        for agent_group_id in agent_groups_id_count.keys():
            obs_dim[agent_group_id] = sample_traj_unbatched.obs[agent_group_id+"_0"].shape[1]
            act_dim[agent_group_id] = get_space_dim(env.action_space(agent_group_id+"_0"))
            agent_net[agent_group_id] = AgentRNN(action_dim=act_dim[agent_group_id], 
                                            hidden_dim=config["AGENT_HIDDEN_DIM"], 
                                            init_scale=config['AGENT_INIT_SCALE'])
            init_x = (
                jnp.zeros((1, 1, obs_dim[agent_group_id])), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
            network_params[agent_group_id] = agent_net[agent_group_id].init(_rng, init_hs, init_x)
            train_state_all[agent_group_id] = TrainState.create(
                apply_fn=agent_net[agent_group_id].apply,
                params=network_params[agent_group_id],
                tx=tx
            )
            target_agent_params[agent_group_id] = jax.tree_map(lambda x: jnp.copy(x),
                                                               train_state_all[agent_group_id].params)
        
        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        def homogeneous_group_pass(params: Any, hidden_state: Any, 
                                   obs: Dict[str, Any], 
                                   dones: Dict[str, Any], 
                                   group_id: str):
            # agents, flatten_agents_obs = [], []
            # for k, v in obs.items():
            #     if k.startswith(gourp_id):
            #         agents.append(k)
            #         flatten_agents_obs.append(v)
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
        
        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, target_agent_params, env_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state
            # EPISODE STEP
            def _env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree_map(lambda x: x[jnp.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[jnp.newaxis, :], last_dones)

                # get the q_values from the agent network
                valid_q_vals_all = {}
                for agent_group_id in agent_groups_id_count.keys():
                    hstate_tmp, q_vals = homogeneous_group_pass(params[agent_group_id], 
                                                                hstate[agent_group_id], 
                                                                obs_, 
                                                                dones_,
                                                                agent_group_id)
                    hstate[agent_group_id] = hstate_tmp
                    # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                    valid_actions = {k:v for k,v in wrapped_env.valid_actions.items() if k.startswith(agent_group_id)}
                    valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], 
                                                          q_vals, 
                                                          valid_actions)
                    valid_q_vals_all.update(valid_q_vals)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals_all, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition

            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = {}
            params = {}
            for agent_group_id in agent_groups_id_count.keys():
                hstate[agent_group_id] = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"],
                                                                     agent_groups_id_count[agent_group_id]*config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
                params[agent_group_id] = train_state[agent_group_id].params
            step_state = (
                params, env_state, init_obs, init_dones, hstate, _rng, time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            # print("before add", traj_batch.actions['agent_0'][0])
            buffer.add_trans(traj_batch.obs, 
                             traj_batch.rewards,  
                             traj_batch.actions,
                             traj_batch.dones,
                             batch_input=True)
            
            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, 
                                          jnp.expand_dims(u, axis=-1), 
                                          axis=-1)
                return jnp.squeeze(q_u, axis=-1)
            
            def compute_target(target_max_qvals, rewards, dones):
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            target_qs,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*(1-done)*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals[-1] * (1-dones[-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (rewards[-2::-1], dones[-2::-1], target_max_qvals[-1::-1])
                    )
                    targets = td_targets[::-1]
                else:
                    # standard DQN loss
                    targets = rewards[:-1] + config["GAMMA"]*(1-dones[:-1])*target_max_qvals
                return targets
            
            def _loss_fn(params, target_agent_params, init_hs, learn_traj, agent_group_id):
                obs_ = {a:learn_traj.obs[a] for a in env.agents if a.startswith(agent_group_id)} # ensure to not pass the global state (obs["__all__"]) to the network
                # q_vals_all = {}
                # target_q_vals_all = {}
                _, q_vals = homogeneous_group_pass(params, 
                                                   init_hs, 
                                                   obs_, 
                                                   learn_traj.dones,
                                                   agent_group_id)
                _, target_q_vals = homogeneous_group_pass(target_agent_params, 
                                                          init_hs, 
                                                          obs_, 
                                                          learn_traj.dones,
                                                          agent_group_id)
                
                # get the q_vals of the taken actions (with exploration) for each agent
                filtered_actions = {k:v for k, v in learn_traj.actions.items() if k.startswith(agent_group_id)}
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    filtered_actions
                )
                # if 'agent_0' in q_vals.keys():
                #     print("filtered_actions", filtered_actions['agent_0'][0])
                #     print('q_vals', q_vals['agent_0'])

                # get the target for each agent (assumes every agent has a reward)
                valid_actions = {k:v for k, v in wrapped_env.valid_actions.items() if k.startswith(agent_group_id)}
                valid_q_vals = jax.tree_util.tree_map(
                    lambda q, valid_idx: q[..., valid_idx],
                    q_vals, 
                    valid_actions
                )
                target_max_qvals = jax.tree_map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
                    target_q_vals,
                    jax.lax.stop_gradient(valid_q_vals)
                )

                # compute a single l2 loss for all the agents in one pass (parameter sharing)
                targets = jax.tree_map(
                    compute_target,
                    target_max_qvals,
                    {agent:learn_traj.rewards[agent] for agent in env.agents if agent.startswith(agent_group_id)}, # rewards and agents could contain additional keys
                    {agent:learn_traj.dones[agent] for agent in env.agents if agent.startswith(agent_group_id)}
                )
                # if 'agent_0' in targets.keys():
                #     print("chosen_action_qvals", chosen_action_qvals['agent_0'])
                #     print("target", targets['agent_0'])
                chosen_action_qvals = jnp.concatenate(list(chosen_action_qvals.values()))
                targets = jnp.concatenate(list(targets.values()))

                if config.get('TD_LAMBDA_LOSS', True):
                    loss = jnp.mean(0.5*((chosen_action_qvals - jax.lax.stop_gradient(targets))**2))
                else:
                    loss = jnp.mean((chosen_action_qvals - jax.lax.stop_gradient(targets))**2)
                
                return loss
            
            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            batch = buffer.sample(rng)
            # batch be like: {'agent_0_obs': xxx, 'agent_0_rew': xxx, ...}
            # trans batch into form: Transition(obs, actions, rewards, dones)
            #   with obs/actions/rewards/dones be like: {'agent_0': xxx, 'agent_1': xxx}
            batch_data_tmp = defaultdict(lambda: defaultdict(dict))
            # print("batch output", batch['agent_0_act'][0])
            for k, v in batch.items():
                if '_' in k:
                    group_id, agent_ind, var_type = k.split('_', 2)
                    agent_id = group_id + '_' + agent_ind
                    batch_data_tmp[var_type][agent_id] = v
                    # batch_data_tmp[var_type][agent_id] = v if var_type is not 'done' else batch_data_tmp["done"]
            learn_traj = Transition(batch_data_tmp['obs'], 
                                    batch_data_tmp['act'],
                                    batch_data_tmp['rew'],
                                    batch_data_tmp['done'],)
            # print("traj output",learn_traj.actions['agent_0'][0])
            loss_all = []
            for agent_group_id in agent_groups_id_count.keys():
                init_hs = ScannedRNN.initialize_carry(
                    config['AGENT_HIDDEN_DIM'],
                    agent_groups_id_count[agent_group_id]*config["BUFFER_BATCH_SIZE"], # (n_agents*batch_size, hs_size)
                ) 
                # compute loss and optimize grad
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
                loss, grads = grad_fn(train_state[agent_group_id].params, 
                                      target_agent_params[agent_group_id], 
                                      init_hs, 
                                      learn_traj,
                                      agent_group_id)
                loss_all.append(loss)
                # print("train with grad", grads['params']['Dense_0']['bias'])
                # apply gradients
                train_state[agent_group_id] = train_state[agent_group_id].apply_gradients(grads=grads)
            
            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}
            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            for agent_group_id in agent_groups_id_count.keys():
                target_agent_params[agent_group_id] = jax.lax.cond(
                    time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                    lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state[agent_group_id].params),
                    lambda _: target_agent_params[agent_group_id],
                    operand=None
                )
            
            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            # test_metrics = {}
            # for agent_group_id in agent_groups_id_count.keys():
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics_all(_rng, train_state, time_state),
                lambda _: test_metrics,
                operand=None
            )
            # print(loss_all)
            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': jnp.mean(jnp.array(loss_all)),
                'rewards': jax.tree_util.tree_map(
                    lambda x: jnp.sum(x, axis=0).mean(), 
                    traj_batch.rewards),
            }
            metrics['test_metrics'] = test_metrics # add the test metrics dictionary

            # logger
            def callback(metrics):
                # info_metrics = {
                #         k:v[...,0][infos["returned_episode"][..., 0]].mean()
                #         for k,v in infos.items() if k!="returned_episode"
                #     }
                logger.add_scalar('Loss/Train', metrics['loss'], metrics['timesteps'])
                for k, v in metrics['rewards'].items():
                    logger.add_scalar(f'Return_{k}/Train', v.mean(), metrics['timesteps'])
                for k, v in metrics['test_metrics'].items():
                    logger.add_scalar(f'{k}/Test', v.mean(), metrics['timesteps'])
            jax.debug.callback(callback, metrics)

            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            return runner_state, metrics

        def get_greedy_metrics_all(rng, train_state, time_state):
            params_all = {}
            for agent_group_id in agent_groups_id_count.keys():
                params_all[agent_group_id] = train_state[agent_group_id].params
            return get_greedy_metrics(rng, params_all, time_state)
        
        def get_greedy_metrics(rng, params, time_state):
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
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
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

            return metrics

        time_state = {'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics_all(_rng, train_state_all, time_state) # initial greedy metrics
        
        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_all,
            target_agent_params,
            env_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # pbar = tqdm(range(config["NUM_UPDATES"]), desc="Training episode")
        # for epoch_i in pbar:
        #     runner_state, metrics = _update_step(runner_state, None)
        return {'runner_state':runner_state, 'metrics':metrics}
    
    return train


if __name__ == "__main__":
    # hyper parameters
    env_batch_size = 8
    save_path = "/home/huaze/enze/jax-mf-vae/jax_ver/baselines/QLearning/checkpoints"
    config = {}
    config["TOTAL_TIMESTEPS"] = 2_050_000
    config["NUM_STEPS"] = 25
    config["NUM_ENVS"] = env_batch_size
    config["NUM_TEST_EPISODES"] = 32
    config['BUFFER_SIZE'] = 5_000
    config['BUFFER_BATCH_SIZE'] = 32
    config["AGENT_HIDDEN_DIM"] = 64
    config['AGENT_INIT_SCALE'] = 2.
    config["NUM_UPDATES"] = 1_00 # 随便写的值
    config["LR"] = 0.001
    config['LR_LINEAR_DECAY'] = False
    config["MAX_GRAD_NORM"] = 25
    config['EPS_ADAM'] = 0.001
    config["EPSILON_START"] = 1.0
    config["EPSILON_FINISH"] = 0.05
    config["EPSILON_ANNEAL_TIME"] = 100_000
    config['TD_LAMBDA_LOSS'] = True
    config['TD_LAMBDA'] = 0.6
    config['GAMMA'] = 0.9
    config["TEST_INTERVAL"] = 50_000
    config["TARGET_UPDATE_INTERVAL"] = 200
    config["NUM_SEEDS"] = 2

    # create env
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
    
    
    # create logger
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / f'iql_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    logger = SummaryWriter(run_dir)

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # train_func = make_train(config, env)
    # outs = train_func(rngs)
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
        flattened_dict = flatten_dict(params, sep=',')
        save_file(flattened_dict, filename)
    model_state = outs['runner_state'][0]
    for k, v in model_state.items():
        params = jax.tree_map(lambda x: x[0], v.params) # save only params of the firt run
        save_dir = os.path.join(save_path, env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{k}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{k}.safetensors')
