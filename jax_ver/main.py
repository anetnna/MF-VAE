import jax
import jax.numpy as jnp

from jax_buffer import JaxFbxBuffer
from model import MAVAE
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

def get_space_size(space_item):
    if isinstance(space_item, gymnax.environments.spaces.Discrete):
        space_shape = space_item.n
        return space_shape
    elif isinstance(space_item, gymnax.environments.spaces.Box):
        return space_item.shape[0]
    else:
        raise NotImplementedError




if __name__ == "__main__":
    # hyper parameters
    ## training parameters
    epoch_num = 256
    sample_num = 128
    batch_size = 128
    train_num = (sample_num // batch_size) * 10
    test_num = 64
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
    env = make('MPE_simple_tag_v3',
                num_good_agents=10,
                num_adversaries=30,
                num_obs=20,)
    agents_id = env.agents
    agent_id_codebook = {}
    for i, agent_id in enumerate(agents_id):
        agent_id_codebook[agent_id] = i
    
    ## create buffer
    buffer = JaxFbxBuffer(max_length=max_size, 
                          min_length=min_size, 
                          batch_size=batch_size, 
                          add_batch=False)
    test_buffer = JaxFbxBuffer(max_length=max_size, 
                               min_length=min_size, 
                               batch_size=batch_size, 
                               add_batch=False)
    # init buffer
    obs, state = env.reset(key_reset)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_act[i]) \
        for i, agent in enumerate(env.agents)}
    # print(f"debug only: @main: {env.action_space(agent_id).n}")
    # print(f"debug only: @main: {actions}")
    next_obs, state, reward, done, infos = env.step(key_step, state, actions)
    buffer.init_buffer(obs, reward, actions, next_obs, done)
    test_buffer.init_buffer(obs, reward, actions, next_obs, done)

    # tmp_buffer = JaxFbxBuffer(max_length=max_size, 
    #                       min_length=min_size, 
    #                       batch_size=batch_size, 
    #                       add_batch=False)
    # tmp_buffer.init_buffer(obs, reward, actions, next_obs, done)
    # for _ in range(batch_size):
    #     tmp_buffer.add_trans(obs, reward, actions, next_obs, done)
    # transitions = tmp_buffer.sample(key_sample)
    # # print(f"debug only: transitions keys {transitions.keys()}")
    # idx_state_all, action_all, rewards, next_states = create_dataset(transitions.experience, agent_id_codebook)

    ## create model
    # gather observation and action dimension
    obs_dim_all = {}
    act_dim_all = {}
    for agent_id in agents_id:
        # print(f"debug only: @main: {env.observation_space(agent_id)}")
        # obs_dim_all[agent_id] = get_space_size(env.observation_space(agent_id))
        obs_dim_all[agent_id] = next_obs[agent_id].shape[0]
        act_dim_all[agent_id] = get_space_size(env.action_space(agent_id))
    # for agent_id in agents_id:
    #     print(f"debug only: @main: {agent_id} with shape {obs_dim_all[agent_id]}")

    # create model with dim information
    model = MAVAE(idx_features=IDX_FEATURES, 
                  obs_features=OBS_FEATURES, 
                  action_features=ACT_FEATURES, 
                  descrete_act=DESCRETE_ACT, 
                  agents=agents_id, 
                  obs_dim=obs_dim_all, 
                  action_dim=act_dim_all)
    fake_idx_state_all = {}
    fake_actions_all = {}
    for agent_id in agents_id:
        fake_state_data = jnp.concatenate([jnp.full((1, ), 0.0), obs[agent_id]], axis=0)
        # print(f"debug only: obs[agent_id] shape: {obs[agent_id].reshape(-1,1).shape}")
        # print(f"debug ony: fake state shape 1: {fake_state_data.shape}")
        # fake_state_data = fake_state_data[jnp.newaxis, :, :]
        # print(f"debug ony: fake state shape 2: {fake_state_data.shape}")
        fake_state_data = fake_state_data[jnp.newaxis, :]
        fake_state_data = jnp.repeat(fake_state_data, repeats=batch_size, axis=0)
        fake_idx_state_all[agent_id] = fake_state_data

        fake_action_data = actions[agent_id][jnp.newaxis, ] # gymnasium.spaces.Discrete
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
                   + "/results") / f'test_huber_loss_ls_{lr}_10_30_20_setup_better_trainer_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    logger = SummaryWriter(run_dir)

    ## main loop
    start_time = time.time()
    pbar = tqdm(range(epoch_num), desc="Training episode")
    for epoch_i in pbar:
        # training epsiode start
        # first sample to fill replay buffer
        for _ in tqdm(range(sample_num), desc="Sample step", leave=False):
            actions = {agent: env.action_space(agent).sample(key_act[i]) \
                       for i, agent in enumerate(env.agents)}
            next_obs, state, reward, done, infos = env.step(key_step, state, actions)
            buffer.add_trans(obs, reward, actions, next_obs, done)
            obs = next_obs
            if True in done.values():
                obs, state = env.reset(key_reset)
            # after filling replay buffer, sample batch to train vae
            # loss_sum = model_trainer.training_model(batched_replay_buffer, train_num, agent_id_codebook)
        loss_sum = 0.
        s_loss_sum = 0.
        r_loss_sum = 0.
        kl_loss_sum = 0.
        for train_setp_i in tqdm(range(train_num), desc="Training vae step", leave=False):
            transitions = buffer.sample(key_sample)
            # print(f"debug only: transitions keys {transitions.keys()}")
            idx_state_all, action_all, rewards, next_states = create_dataset(transitions.experience, agent_id_codebook)
            # train_step_i = epoch_i * train_num + step_i
            # train_state_old = train_state
            train_state, loss, s_loss, r_loss, kl_loss = train_step(train_state, 
                                                                    idx_state_all, 
                                                                    action_all, 
                                                                    next_states, 
                                                                    rewards, 
                                                                    key_train,)
            # print(train_state_old.params['Encoder_0']['Dense_0']['bias'])
            # print(f"debug ony: @main: step: {train_setp_i}, epoch: {epoch_i}, param: {train_state.params['Encoder_0']['Dense_0']['bias']}")
            # exit()
            loss_sum += loss
            s_loss_sum += s_loss
            r_loss_sum += r_loss
            kl_loss_sum += kl_loss
        loss_sum /= train_num * 1.0
        s_loss_sum /= train_num * 1.0
        r_loss_sum /= train_num * 1.0
        kl_loss_sum /= train_num * 1.0
        logger.add_scalar('Loss/Train', loss_sum, epoch_i)
        logger.add_scalar('Loss/State_Train', s_loss, epoch_i)
        logger.add_scalar('Loss/Reward_Train', r_loss,epoch_i)
        logger.add_scalar('Loss/KL_Train', kl_loss, epoch_i)

        for _ in tqdm(range(sample_num), desc="Test sample step", leave=False):
            actions = {agent: env.action_space(agent).sample(key_act[i]) \
                       for i, agent in enumerate(env.agents)}
            next_obs, state, reward, done, infos = env.step(key_step, state, actions)
            test_buffer.add_trans(obs, reward, actions, next_obs, done)
            obs = next_obs
            if True in done.values():
                obs, state = env.reset(key_reset)
        
        loss_sum = 0.
        s_loss_sum = 0.
        r_loss_sum = 0.
        kl_loss_sum = 0.
        for train_setp_i in tqdm(range(test_num), desc="Test vae step", leave=False):
            transitions = test_buffer.sample(key_sample)
            # print(f"debug only: transitions keys {transitions.keys()}")
            idx_state_all, action_all, rewards, next_states = create_dataset(transitions.experience, agent_id_codebook)
            # train_step_i = epoch_i * train_num + step_i
            # train_state_old = train_state
            loss, s_loss, r_loss, kl_loss = test_step(train_state, 
                                                      idx_state_all, 
                                                      action_all, 
                                                      next_states, 
                                                      rewards, 
                                                      key_train,)
            # print(train_state_old.params['Encoder_0']['Dense_0']['bias'])
            # print(f"debug ony: @main: step: {train_setp_i}, epoch: {epoch_i}, param: {train_state.params['Encoder_0']['Dense_0']['bias']}")
            # exit()
            loss_sum += loss
            s_loss_sum += s_loss
            r_loss_sum += r_loss
            kl_loss_sum += kl_loss
        loss_sum /= train_num * 1.0
        s_loss_sum /= train_num * 1.0
        r_loss_sum /= train_num * 1.0
        kl_loss_sum /= train_num * 1.0
        logger.add_scalar('Loss/Test', loss_sum, epoch_i)
        logger.add_scalar('Loss/State_Test', s_loss, epoch_i)
        logger.add_scalar('Loss/Reward_Test', r_loss,epoch_i)
        logger.add_scalar('Loss/KL_Test', kl_loss, epoch_i)
    
    end_time = time.time()

    with open('./model_save/vae/model_state.pkl', 'wb') as f:
        pickle.dump(train_state.params, f)
    
    # print(f"obs shape: {observations['adversary_0'].shape}")
    print(end_time - start_time)

