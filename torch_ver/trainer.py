import torch
import os
import numpy as np
from tqdm import tqdm


def create_dataset(transition, codebook):
    idx_state_all = {}
    action_all = {}
    # next_state_rew_all = None
    rewards = None
    next_states = None
    for agent_id in codebook.keys():
        agent_id_num = codebook[agent_id]
        # print(f"debug only: @create dataset: obs shape: {np.shape(agent_id_num)}")
        obs = transition[agent_id + "_observations"]
        # print(f"debug only: @create dataset: obs shape: {np.shape(obs)}")
        action = transition[agent_id + "_actions"]
        next_obs = transition[agent_id + "_next_observations"]
        rew = transition[agent_id + "_rewards"]
        idx_state_all[agent_id] = torch.from_numpy(np.insert(obs, 0, agent_id_num, axis=1))
        action_all[agent_id] = torch.from_numpy(action)
        if rewards is not None:
            rewards = np.concatenate((rewards, rew), axis=1)
        else:
            rewards = rew
        if next_states is not None:
            next_states = np.concatenate((next_states, next_obs), axis=1)
        else:
            next_states = next_obs
        # if next_state_rew_all is not None:
        #     next_state_rew_all = np.concatenate((next_state_rew_all, np.concatenate((next_obs, rew), axis=1)), axis=1)
        # else:
        #     next_state_rew_all = np.concatenate((next_obs, rew), axis=1)
    # idx_state_all_np = np.array(idx_state_all)
    # action_all_np = np.array(action_all)
    # next_state_rew_all_np = np.array(next_state_rew_all)
    # shape_next_state_rew_all = np.shape(next_state_rew_all_np)
    # next_state_rew_all_np = next_state_rew_all_np.reshape((shape_next_state_rew_all[0], -1))
    next_state_rew_all = np.concatenate((next_states, rewards), axis=1)
    next_state_rew_all_torch = torch.from_numpy(next_state_rew_all)

    next_states_torch = torch.from_numpy(next_states)
    rewards_torch = torch.from_numpy(rewards)
    return idx_state_all, action_all, next_state_rew_all_torch, next_states_torch, rewards_torch
    

class Trainer:
    def __init__(self, mode, model, lr, loss_func, beta=None, device='cuda:0'):
        self.mode = mode
        assert self.mode in ['Adam', 'ART', 'POPART']
        self.model = model
        self.sigma = torch.tensor(1., dtype=torch.float).to(device)
        self.sigma_new = None
        self.mu = torch.tensor(0., dtype=torch.float).to(device)
        self.mu_new = None
        self.nu = self.sigma**2 + self.mu**2 # second order moment
        self.beta = beta
        self.lr = lr
        self.loss_func = loss_func
        self.loss = None
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
        self.device = device
        # self.opt_upper = torch.optim.Adam
    
    def art(self, y):
        self.mu_new = (1. - self.beta) * self.mu + self.beta * y.to(self.device)
        self.nu = (1. - self.beta) * self.nu + self.beta * y.to(self.device)**2
        self.sigma_new = torch.sqrt(self.nu - self.mu_new**2)
    
    def pop(self):
        relative_sigma = (self.sigma / self.sigma_new)
        self.model.reward_linear.weight.data.mul_(relative_sigma)
        self.model.reward_linear.bias.data.mul_(relative_sigma).add_((self.mu-self.mu_new)/self.sigma_new)
    
    def update_stats(self):
        if self.sigma_new is not None:
            self.sigma = self.sigma_new
        if self.mu_new is not None:
            self.mu = self.mu_new
    
    def normalize(self, y):
        return (y.to(self.device) - self.mu) / self.sigma
    
    def denormalize(self, y):
        return self.sigma * y.to(self.device) + self.mu
    
    def forward(self, idx_state, actions, s_hat, r_hat):
        if self.mode in ['POPART', 'ART']:
            self.art(r_hat)
        if self.mode in ['POPART']:
            self.pop()
        self.update_stats()
        recon_s, recon_r, mean_all, logvar_all = self.model(idx_state, actions)
        self.loss, _, _, _ = self.loss_func(recon_s, recon_r, s_hat, self.normalize(r_hat), mean_all, logvar_all, self.device)
        return recon_s, recon_r, mean_all, logvar_all
    
    def backward(self):
        self.opt.zero_grad()
        self.loss.backward()
    
    def step(self):
        self.opt.step()
    
    def training_model(self, batched_replay_buffer, train_num, agent_id_codebook):
        loss_train = 0.0
        pbar = tqdm(range(train_num), desc="Training vae step", leave=False)
        for train_step_i in pbar:
            transitions = batched_replay_buffer.sample()
            idx_state, actions, next_state_rew, next_state, rewards = create_dataset(transitions, agent_id_codebook)
            recon_s, recon_r, mean_all, logvar_all = self.forward(idx_state, actions, next_state, rewards)
            loss_train_step = self.loss_func(recon_s, self.denormalize(recon_r), next_state, rewards, mean_all, logvar_all, self.device)
            loss_train += loss_train_step
            # one step training
            self.backward()
            self.step()
            pbar.set_description(f"epoch {train_step_i}: train loss {loss_train_step:.5f}.")
        loss_train /= (train_num * 1.0)
        return loss_train

