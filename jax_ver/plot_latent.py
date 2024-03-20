import jax
import jax.numpy as jnp
import numpy as np

from src.jax_buffer import JaxFbxBuffer
from src.env import get_space_dim, EnvRolloutManager
from model import MAVAE
from jaxmarl import make
from trainer import create_dataset, train_step, test_step
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pickle
from flax.training.train_state import TrainState
from flax.core import freeze
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


def load_jax_params(model_path):
    with open(model_path, 'rb') as f:
        jax_params = pickle.load(f)
    return jax_params


if __name__ == "__main__":
    # hyper parameters
    ## training parameters
    env_batch_size = 8
    sample_num = 128 // env_batch_size
    max_size = 10_000
    min_size = 64
    batch_size = 128

    ## model parameters
    IDX_FEATURES = 64
    OBS_FEATURES = 64
    ACT_FEATURES = 64
    DESCRETE_ACT = True
    devices = jax.devices()

    ## init random number generator
    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step, key_sample, key_model, key_train = jax.random.split(key, 7)

    # create env
    plain_env = make('MPE_simple_tag_v3',
                num_good_agents=10,
                num_adversaries=30,
                num_obs=20,)
    env = EnvRolloutManager(plain_env, batch_size=env_batch_size)
    obs, state = env.batch_reset(key_reset)
    agents_id = env.agents
    agent_id_codebook = {}
    for i, agent_id in enumerate(agents_id):
        agent_id_codebook[agent_id] = i
    
    # create buffer
    buffer = JaxFbxBuffer(max_length=max_size, 
                          min_length=min_size, 
                          batch_size=batch_size, 
                          add_batch=True)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {agent: env.batch_sample(key_act[i], agent) \
        for i, agent in enumerate(env.agents)}
    next_obs, state, reward, done, infos = env.batch_step(key_step, state, actions)
    
    obs_unbatched = jax.tree_map(lambda x: x[0, :], obs)
    reward_unbatched = jax.tree_map(lambda x: x[0, ], reward)
    actions_unbatched = jax.tree_map(lambda x: x[0], actions)
    next_obs_unbatched = jax.tree_map(lambda x: x[0, :], next_obs)
    done_unbatched = jax.tree_map(lambda x: x[0], done)
    buffer.init_buffer(obs_unbatched, reward_unbatched, actions_unbatched, next_obs_unbatched, done_unbatched)

    # load model
    ## parameters
    model_path = "/home/huaze/enze/jax-mf-vae/jax_ver/model_save/vae/model_batch_state.pkl"
    jax_params = load_jax_params(model_path)
    jax_params = freeze(jax_params)

    ## create model
    obs_dim_all = {}
    act_dim_all = {}
    obs_unbatched = jax.tree_map(lambda x: x[0, :], obs)
    for agent_id in agents_id:
        obs_dim_all[agent_id] = obs_unbatched[agent_id].shape[0]
        act_dim_all[agent_id] = get_space_dim(env.action_space(agent_id))
    
    model = MAVAE(idx_features=IDX_FEATURES, 
                  obs_features=OBS_FEATURES, 
                  action_features=ACT_FEATURES, 
                  descrete_act=DESCRETE_ACT, 
                  agents=agents_id, 
                  obs_dim=obs_dim_all, 
                  action_dim=act_dim_all)
    
    buffer = sample_from_env(env, buffer, sample_num, "Sample steps", key_reset)
    transitions = buffer.sample(key_sample)
    idx_state_all, _, _, _ = create_dataset(transitions.experience, agent_id_codebook)

    output_mu, output_logvar = model.apply({'params': jax_params}, idx_state_all, key_train, method=model.output_latent)
    print(f"mu shape: {output_mu.shape}")
    print(f"logvar shape: {output_logvar.shape}")
    
    mu_reduced = TSNE(n_components=2).fit_transform(output_mu)

    agent_index = []
    for i in range(env.num_agents):
        agent_index += [i]*batch_size

    markers = ['o', 's']  # 'o' 和 's' 分别表示圆形和方形
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    plt.figure(figsize=(8, 6))
    # plt.scatter(mu_reduced[:, 0], mu_reduced[:, 1], c=agent_index, alpha=0.5)
    for i in range(env.num_agents):
        plt.scatter(mu_reduced[i*batch_size:(i+1)*batch_size, 0], mu_reduced[i*batch_size:(i+1)*batch_size, 1], color=colors[i % 20], marker=markers[i // 20])

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Mu Latent Variables')
    plt.savefig("./model_save/result_figs/mu.png")
    
    logvar_reduced = TSNE(n_components=2).fit_transform(output_logvar)

    plt.figure(figsize=(8, 6))
    for i in range(env.num_agents):
        idx = agent_index == i
        plt.scatter(logvar_reduced[i*batch_size:(i+1)*batch_size, 0], logvar_reduced[i*batch_size:(i+1)*batch_size, 1], color=colors[i % 20], marker=markers[i // 20])
    # plt.scatter(logvar_reduced[:, 0], logvar_reduced[:, 1], c=agent_index, alpha=0.5)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Log Var Latent Variables')
    plt.savefig("./model_save/result_figs/var.png")
