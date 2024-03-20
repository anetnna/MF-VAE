import jax
import jax.numpy as jnp

from src.jax_buffer import JaxFbxBuffer
from src.env import get_space_dim, EnvRolloutManager
from model import MAVAE, MAVAEAtten
from jaxmarl import make
from trainer import create_dataset, train_step, test_step
from tqdm import tqdm

import pickle
from flax.training.train_state import TrainState
import optax

from tensorboardX import SummaryWriter

from pathlib import Path
import os

import time
from datetime import datetime
import gymnax
from typing import Any, Dict, Tuple


def sample_from_env(env: EnvRolloutManager, 
                    sample_buffer: JaxFbxBuffer, 
                    sample_steps: int, 
                    desc: str, 
                    key_sample_env: Any)->JaxFbxBuffer:
    obs, state = env.batch_reset(key_sample_env)
    for _ in tqdm(range(sample_steps), desc=desc, leave=False):
        actions = {agent: env.batch_sample(key_act[i], agent) \
                    for i, agent in enumerate(env.agents)}
        next_obs, state, reward, done, infos = env.batch_step(key_step, state, actions)
        sample_buffer.add_trans(obs, reward, actions, next_obs, done, batch_input=True)
        obs = next_obs
        # print(done.values())
        if any(done_array.any() for done_array in done.values()):
            obs, state = env.batch_reset(key_reset)
    return sample_buffer


def apply_network(run_steps: int, 
                  desc: str, 
                  transition_buffer: JaxFbxBuffer, 
                  agent_id_codebook: Dict[str, Any],
                  task: str,
                  train_net: Any,
                  multi_agent_output: bool=False) -> Tuple[float, float, float, float, Any]:
    if task not in ["train", "test"]:
        raise ValueError("task must be train or test")
    loss_sum = 0.
    s_loss_sum = 0.
    r_loss_sum = 0.
    kl_loss_sum = 0.
    for _ in tqdm(range(run_steps), desc=desc, leave=False):
        transitions = transition_buffer.sample(key_sample)
        # print(f"debug only: transitions keys {transitions.keys()}")
        idx_state_all, action_all, rewards, next_states = create_dataset(transitions.experience, 
                                                                         agent_id_codebook,
                                                                         multi_agent_output)
        if task == "train":
            train_net, loss, s_loss, r_loss, kl_loss = train_step(train_net, 
                                                                  idx_state_all, 
                                                                  action_all, 
                                                                  next_states, 
                                                                  rewards, 
                                                                  key_train,)
        else:
            loss, s_loss, r_loss, kl_loss = test_step(train_net, 
                                                      idx_state_all, 
                                                      action_all, 
                                                      next_states, 
                                                      rewards, 
                                                      key_train,)
        loss_sum += loss
        s_loss_sum += s_loss
        r_loss_sum += r_loss
        kl_loss_sum += kl_loss
    loss_sum /= train_num * 1.0
    s_loss_sum /= train_num * 1.0
    r_loss_sum /= train_num * 1.0
    kl_loss_sum /= train_num * 1.0
    return loss_sum, s_loss_sum, r_loss_sum, kl_loss_sum, train_net



if __name__ == "__main__":
    # hyper parameters
    ## training parameters
    epoch_num = 256
    batch_size = 128
    env_batch_size = 8
    sample_num = 128 // env_batch_size
    train_num = ((sample_num * env_batch_size) // batch_size) * 10
    test_num = train_num
    max_size = 10_000
    min_size = 64
    lr = 0.001

    ## model parameters
    IDX_FEATURES = 64
    OBS_FEATURES = 64
    ACT_FEATURES = 64
    DESCRETE_ACT = True
    devices = jax.devices()

    ## init random number generator
    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step, key_sample, key_model, key_train = jax.random.split(key, 7)

    ## create env
    plain_env = make('MPE_simple_tag_v3',
                num_good_agents=10,
                num_adversaries=30,
                num_obs=20,)
    env = EnvRolloutManager(plain_env, batch_size=env_batch_size)
    agents_id = env.agents
    agent_id_codebook = {}
    for i, agent_id in enumerate(agents_id):
        agent_id_codebook[agent_id] = i
    
    ## create buffer
    buffer = JaxFbxBuffer(max_length=max_size, 
                          min_length=min_size, 
                          batch_size=batch_size, 
                          add_batch=True)
    test_buffer = JaxFbxBuffer(max_length=max_size, 
                               min_length=min_size, 
                               batch_size=batch_size, 
                               add_batch=True)
    # init buffer
    obs, state = env.batch_reset(key_reset)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {agent: env.batch_sample(key_act[i], agent) \
        for i, agent in enumerate(env.agents)}
    # print(f"debug only: @main: {env.action_space(agent_id).n}")
    # print(f"debug only: @main: {actions}")
    next_obs, state, reward, done, infos = env.batch_step(key_step, state, actions)
    
    obs_unbatched = jax.tree_map(lambda x: x[0, :], obs)
    reward_unbatched = jax.tree_map(lambda x: x[0, ], reward)
    actions_unbatched = jax.tree_map(lambda x: x[0], actions)
    next_obs_unbatched = jax.tree_map(lambda x: x[0, :], next_obs)
    done_unbatched = jax.tree_map(lambda x: x[0], done)
    buffer.init_buffer(obs_unbatched, reward_unbatched, actions_unbatched, next_obs_unbatched, done_unbatched)
    test_buffer.init_buffer(obs_unbatched, reward_unbatched, actions_unbatched, next_obs_unbatched, done_unbatched)

    ## create model
    # gather observation and action dimension
    obs_dim_all = {}
    act_dim_all = {}
    for agent_id in agents_id:
        # print(f"debug only: @main: {env.observation_space(agent_id)}")
        # obs_dim_all[agent_id] = get_space_size(env.observation_space(agent_id))
        obs_dim_all[agent_id] = next_obs_unbatched[agent_id].shape[0]
        act_dim_all[agent_id] = get_space_dim(env.action_space(agent_id))
    # for agent_id in agents_id:
    #     print(f"debug only: @main: {agent_id} with shape {obs_dim_all[agent_id]}")

    # create model with dim information
    using_attention = True
    if not using_attention:
        model = MAVAE(idx_features=IDX_FEATURES, 
                        obs_features=OBS_FEATURES, 
                        action_features=ACT_FEATURES, 
                        descrete_act=DESCRETE_ACT, 
                        agents=agents_id, 
                        obs_dim=obs_dim_all, 
                        action_dim=act_dim_all)
    else:
        model = MAVAEAtten(idx_features=IDX_FEATURES, 
                        obs_features=OBS_FEATURES, 
                        action_features=ACT_FEATURES, 
                        descrete_act=DESCRETE_ACT, 
                        agents=agents_id, 
                        obs_dim=obs_dim_all, 
                        action_dim=act_dim_all)
    fake_idx_state_all = {}
    fake_actions_all = {}
    for agent_id in agents_id:
        fake_state_data = jnp.concatenate([jnp.full((1, ), 0.0), obs_unbatched[agent_id]], axis=0)
        # print(f"debug only: obs[agent_id] shape: {obs[agent_id].reshape(-1,1).shape}")
        # print(f"debug ony: fake state shape 1: {fake_state_data.shape}")
        # fake_state_data = fake_state_data[jnp.newaxis, :, :]
        # print(f"debug ony: fake state shape 2: {fake_state_data.shape}")
        fake_state_data = fake_state_data[jnp.newaxis, :]
        fake_state_data = jnp.repeat(fake_state_data, repeats=batch_size, axis=0)
        fake_idx_state_all[agent_id] = fake_state_data

        fake_action_data = actions_unbatched[agent_id][jnp.newaxis, ] # gymnasium.spaces.Discrete
        fake_action_data = jnp.repeat(fake_action_data, repeats=batch_size, axis=0)
        fake_actions_all[agent_id] = fake_action_data
        # print(f"debug ony: @main: fake state shape: {fake_state_data.shape}")
        # print(f"debug ony: @main: {agent_id} fake action: {fake_action_data}")

    params = model.init(key_model, fake_idx_state_all, fake_actions_all, key_model)['params']
    
    # create optimizer
    optimizer = optax.adam(learning_rate=lr)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # create logger
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / f'atten_ls_{lr}_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    logger = SummaryWriter(run_dir)

    ## main loop
    start_time = time.time()
    pbar = tqdm(range(epoch_num), desc="Training episode")
    for epoch_i in pbar:
        # training epsiode start
        # first sample to fill replay buffer
        buffer = sample_from_env(env, buffer, sample_num, "Sample steps", key_reset)
        loss_sum, s_loss_sum, r_loss_sum, kl_loss_sum, train_state = apply_network(train_num, 
                                                                                   "Training VAE steps",
                                                                                   buffer,
                                                                                   agent_id_codebook,
                                                                                   "train",
                                                                                   train_state,
                                                                                   using_attention)

        logger.add_scalar('Loss/Train', loss_sum, epoch_i)
        logger.add_scalar('Loss/State_Train', s_loss_sum, epoch_i)
        logger.add_scalar('Loss/Reward_Train', r_loss_sum,epoch_i)
        logger.add_scalar('Loss/KL_Train', kl_loss_sum, epoch_i)

        test_buffer = sample_from_env(env, test_buffer, sample_num, "Test sample steps", key_reset)
        loss_sum, s_loss_sum, r_loss_sum, kl_loss_sum, train_state = apply_network(test_num, 
                                                                                   "Test VAE step",
                                                                                   test_buffer,
                                                                                   agent_id_codebook,
                                                                                   "test",
                                                                                   train_state,
                                                                                   using_attention)
        
        logger.add_scalar('Loss/Test', loss_sum, epoch_i)
        logger.add_scalar('Loss/State_Test', s_loss_sum, epoch_i)
        logger.add_scalar('Loss/Reward_Test', r_loss_sum,epoch_i)
        logger.add_scalar('Loss/KL_Test', kl_loss_sum, epoch_i)
    
    end_time = time.time()

    with open('./model_save/vae/model_batch_state_right_atten.pkl', 'wb') as f:
        pickle.dump(train_state.params, f)
    
    # print(f"obs shape: {observations['adversary_0'].shape}")
    print(end_time - start_time)

