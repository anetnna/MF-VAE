import torch
import torch.nn as nn
from torch import optim
import numpy as np
from src.env import create_transition, create_env
from src.replay_buffer import MultiAgentCPPRB
from model import MAVAE, loss_vae_fn, loss_s_r_vae_fn
from trainer import create_dataset, Trainer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os

import time



if __name__ == "__main__":
    # hyper parameters
    ## training parameters
    epoch_num = 128
    sample_num = 128
    batch_size = 128
    train_num = (sample_num // batch_size) * 4
    test_num = 128
    max_size = 10000
    lr = 0.005

    ## model parameters
    IDX_FEATURES = 64
    OBS_FEATURES = 64
    ACT_FEATURES = 64
    DESCRETE_ACT = True
    device = 'cuda:0'

    # create environment
    parallel_env, obs_dim_all, act_dim_all, obs, infos = create_env('simple_tag_v3')
    # print(f"debug only: action_dim: {act_dim_all}")
    agents_id = parallel_env.agents
    agent_id_codebook = {}
    for i, agent_id in enumerate(agents_id):
        agent_id_codebook[agent_id] = i

    # create replay buffer
    batched_replay_buffer = MultiAgentCPPRB(parallel_env, max_size=max_size, batch_size=batch_size)

    # create model
    model = MAVAE(IDX_FEATURES, OBS_FEATURES, ACT_FEATURES, DESCRETE_ACT, agents_id, obs_dim_all, act_dim_all, device)
    model.to(device)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
    model_trainer = Trainer('POPART', model, lr, loss_s_r_vae_fn, beta=0.0, device=device)

    # create logger
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / 'test_huber_loss'
    logger = SummaryWriter(run_dir)

    # main loop
    start_time = time.time()
    pbar = tqdm(range(epoch_num), desc="Training episode")
    for epoch_i in pbar:
        # training epsiode start
        # first sample to fill replay buffer
        for _ in tqdm(range(sample_num), desc="Sample step", leave=False):
            # this is where you would insert your policy
            actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
            next_obs, rewards, terminations, truncations, infos = parallel_env.step(actions)
            # print(f"debug only: @sample: rewards info: {rewards}")
            batched_replay_buffer.add(obs, next_obs, actions, rewards, terminations, truncations)
            obs = next_obs
            if True in terminations.values() or True in truncations.values():
                obs, infos = parallel_env.reset(seed=42)
                batched_replay_buffer.on_episode_end()
        # after filling replay buffer, sample batch to train vae
        # loss_sum = model_trainer.training_model(batched_replay_buffer, train_num, agent_id_codebook)
        loss_sum = 0.
        s_loss_sum = 0.
        r_loss_sum = 0.
        kl_loss_sum = 0.
        for _ in tqdm(range(train_num), desc="Training vae step", leave=False):
            transitions = batched_replay_buffer.sample()
            idx_state, actions, next_state_rew, next_state, rewards = create_dataset(transitions, agent_id_codebook)
            # recon, mu_all, logvar_all = model(idx_state, actions)
            recon_state, recon_reward, mu_all, logvar_all = model(idx_state, actions)
            # print(f"debug only: @forward: recon_state shape: {recon_state.shape}, recon_reward shape: {recon_reward.shape}, next_state_rew shape: {next_state_rew.shape}")
            # compute loss and backward
            # loss = loss_vae_fn(recon, next_state_rew, mu_all, logvar_all, device)
            loss, s_loss, r_loss, kl_loss = loss_s_r_vae_fn(recon_state, recon_reward, next_state, rewards, mu_all, logvar_all, device)
            optimizer.zero_grad()
            loss.backward()
            # for name, param in model.named_parameters():
            #     print(name, param.grad)
            # exit()
            optimizer.step()
            scheduler.step()
            loss_sum += loss
            s_loss_sum += s_loss
            r_loss_sum += r_loss
            kl_loss_sum += kl_loss
        loss_sum /= train_num * 1.0
        pbar.set_description(f"epoch {epoch_i}: train loss {loss_sum:.5f}.")
        logger.add_scalar('Loss/Train', loss_sum, epoch_i)
        logger.add_scalar('Loss/State_Train', s_loss_sum, epoch_i)
        logger.add_scalar('Loss/Reward_Train', r_loss_sum, epoch_i)
        logger.add_scalar('Loss/KL_Train', kl_loss_sum, epoch_i)

    # save model
    save_path = './model_save/test.pt'    
    model.save(save_path)      

    # print(f"idx_state_shape: {np.shape(idx_state)}, actions shape: {np.shape(actions)}, next_state_rew shape: {np.shape(next_state_rew)}")
    parallel_env.close()
    print(transitions['agent_0_observations'].shape)
    end_time = time.time()
    
    # print(f"obs shape: {observations['adversary_0'].shape}")
    print(end_time - start_time)
