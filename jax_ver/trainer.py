import jax
import jax.numpy as jnp
from jax import random, vmap
from tqdm import tqdm

from typing import Dict, Any, Optional, Tuple


def create_dataset(transition: Dict[str, Any], 
                   codebook: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], jnp.ndarray, jnp.ndarray]:
    idx_state_all = {}
    action_all = {}
    # rewards = None
    # next_states = None
    for agent_id in codebook.keys():
        agent_id_num = codebook[agent_id]
        # print(f"debug only: @create dataset: obs shape: {np.shape(agent_id_num)}")
        obs = transition[agent_id + "_obs"]
        # print(f"debug only: @create dataset: obs shape: {np.shape(obs)}")
        action = transition[agent_id + "_act"]
        # next_obs = transition[agent_id + "_next_obs"]
        # rew = transition[agent_id + "_rew"]
        idx_state_all[agent_id] = jnp.squeeze(jnp.concatenate([jnp.full((obs.shape[0], 1, 1), agent_id_num), obs], axis=1))
        action_all[agent_id] = jnp.squeeze(action)
    # print(f"debug only: @dataset: rewards_list {transition['adversary_0_rew'].shape}")
    rewards_list = [jnp.squeeze(v).reshape(-1, 1) for k, v in transition.items() if k.endswith("_rew")]
    # print(f"debug only: @dataset: rewards_list {rewards_list}")
    rewards = jnp.concatenate(rewards_list, axis=1)
    next_states_list  = [jnp.squeeze(v) for k, v in transition.items() if k.endswith("_next_obs")]
    next_states = jnp.concatenate(next_states_list, axis=1)

    # print(f"debug only: @create_dataet: idx_state_all shape: {idx_state_all[agent_id].shape}")
    # print(f"debug only: @create_dataet: action_all shape: {action_all[agent_id].shape}")
    
    # print(f"debug only: @create_dataet: rewards shape: {rewards.shape}")
    # print(f"debug only: @create_dataet: next_states shape: {next_states.shape}")
    # print(f"debug only: @create_dataset: next state: {transition["agent_7_next_obs"]}")

    return idx_state_all, action_all, rewards, next_states


kl_weight = 0.1
r_weight = 0.5

def mse_loss(x, y):
    return jnp.mean((x - y) ** 2)

def huber_loss(x, y, delta=1.0):
    abs_error = jnp.abs(x - y)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic ** 2 + delta * linear)

def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)

def loss_s_r_vae_fn(recon_s, recon_r, s_hat, r_hat, mean_all, logvar_all, using_huber_loss=True):
    # Note: No need to move tensors to a device in JAX.
    s_loss_fn = huber_loss if using_huber_loss else mse_loss
    r_loss_fn = huber_loss if using_huber_loss else mse_loss

    s_loss = s_loss_fn(s_hat, recon_s)
    r_loss = r_loss_fn(r_hat, recon_r)
    recons_loss = s_loss * (1-r_weight) + r_loss * r_weight

    # print(f"debug only: @loss fn: mean_all {len(mean_all)}, logvar_all: {len(logvar_all)}")
    # mean_all, logvar_all = jnp.array(mean_all), jnp.array(logvar_all)
    kl_loss = jnp.mean(vmap(kl_divergence)(mean_all, logvar_all))
    
    loss = recons_loss + kl_loss * kl_weight
    return loss, s_loss, r_loss, kl_loss

@jax.jit
def train_step(train_state, idx_state, actions, next_state, rewards, rng_key):
    def loss_fn(params):
        recon_state, recon_reward, mu_all, logvar_all = train_state.apply_fn({'params': params}, idx_state, actions, rng_key)
        loss, s_loss, r_loss, kl_loss = loss_s_r_vae_fn(recon_state, recon_reward, next_state, rewards, mu_all, logvar_all)
        return loss
    
    recon_state, recon_reward, mu_all, logvar_all = train_state.apply_fn({'params': train_state.params}, idx_state, actions, rng_key)
    # print(f"debug only: @train step: {next_state.shape}")
    loss, s_loss, r_loss, kl_loss = loss_s_r_vae_fn(recon_state, recon_reward, next_state, rewards, mu_all, logvar_all) 
    grads = jax.grad(loss_fn)(train_state.params)
    return train_state.apply_gradients(grads=grads), loss, s_loss, r_loss, kl_loss

@jax.jit
def test_step(train_state, idx_state, actions, next_state, rewards, rng_key):
    recon_state, recon_reward, mu_all, logvar_all = train_state.apply_fn({'params': train_state.params}, idx_state, actions, rng_key)
    loss, s_loss, r_loss, kl_loss = loss_s_r_vae_fn(recon_state, recon_reward, next_state, rewards, mu_all, logvar_all)
    return loss, s_loss, r_loss, kl_loss
