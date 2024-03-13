import torch
import torch.nn as nn
import torch.nn.functional as F

kl_weight = 0.0025
r_weight = 0.005

def loss_vae_fn(y, y_hat, mean_all, logvar_all, device):
    y = y.to(device)
    y_hat = y_hat.to(device)
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = 0.0
    for i in range(len(mean_all)):
        kl_loss += torch.mean(-0.5 * torch.sum(1 + logvar_all[i] - mean_all[i]**2 - torch.exp(logvar_all[i]), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss


def loss_s_r_vae_fn(recon_s, recon_r, s_hat, r_hat, mean_all, logvar_all, device, using_huber_loss=True):
    recon_s = recon_s.to(device)
    recon_r = recon_r.to(device)
    s_hat = s_hat.to(device)
    r_hat = r_hat.to(device)
    # s_loss = F.mse_loss(s_hat, recon_s)
    if using_huber_loss:
        s_loss = F.huber_loss(s_hat, recon_s)
    else:
        s_loss = F.mse_loss(s_hat, recon_s)
    # r_loss = F.mse_loss(r_hat, recon_r)
    if using_huber_loss:
        r_loss = F.huber_loss(r_hat, recon_r)
    else:
        r_loss = F.mse_loss(r_hat, recon_r)
    recons_loss = s_loss + r_loss * r_weight
    kl_loss = 0.0
    for i in range(len(mean_all)):
        kl_loss += torch.mean(-0.5 * torch.sum(1 + logvar_all[i] - mean_all[i]**2 - torch.exp(logvar_all[i]), 1), 0)
    # print(f"debug only: @loss compute: s_loss: {s_loss}, r_loss: {r_loss}, kl_loss: {kl_loss}")
    loss = recons_loss + kl_loss * kl_weight
    return loss, s_loss, r_loss, kl_loss


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        HIDDEN = [64, 64, 256]
        layers = []
        in_features = in_dim
        for hidden_dim in HIDDEN:
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features=in_features, out_features=out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ActionEncoder, self).__init__()
        HIDDEN = [64]
        layers = []
        in_features = in_dim
        for hidden_dim in HIDDEN:
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features=in_features, out_features=out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        HIDDEN = [1024, 256, 64, 256, 1024]
        layers = []
        in_features = in_dim
        for hidden_dim in HIDDEN:
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features=in_features, out_features=out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MAVAE(nn.Module):
    def __init__(self, idx_features: int, obs_features: int, action_features: int, descrete_act: bool, agents: list, obs_dim: dict, action_dim: dict, device: str):
        super(MAVAE, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = action_dim
        self.feature = obs_features

        self.device = device
        self.descrete_act = descrete_act

        # encoder
        self.encoders = {}
        self.idx_emb = nn.Embedding(len(agents), idx_features).to(self.device)
        self.action_encoder = {}

        reward_out_dim = len(agents)
        obs_out_dim = 0
        for agent_id in agents:
            self.encoders[agent_id] = Encoder(self.obs_dim[agent_id] + idx_features, 2*obs_features).to(self.device)
            if descrete_act:
                self.action_encoder[agent_id] = nn.Embedding(action_dim[agent_id], action_features).to(self.device)
            else:
                self.action_encoder[agent_id] = ActionEncoder(self.act_dim[agent_id], action_features).to(self.device)
            obs_out_dim += obs_dim[agent_id]
        decoder_out_dim = reward_out_dim + obs_out_dim

        self.decoder = Decoder(in_dim=(obs_features + action_features)*len(agents), out_dim=decoder_out_dim).to(self.device)
        self.state_decoder = Decoder(in_dim=(obs_features + action_features)*len(agents), out_dim=obs_out_dim).to(self.device)
        self.reward_decoder = Decoder(in_dim=(obs_features + action_features)*len(agents), out_dim=reward_out_dim).to(self.device)
        self.reward_linear = nn.Linear(in_features=reward_out_dim, out_features=reward_out_dim)
        torch.nn.init.ones_(self.reward_linear.weight)
        torch.nn.init.zeros_(self.reward_linear.bias)
    
    def forward(self, idx_state, actions):
        z_all = []
        mu_all = []
        log_var_all = []
        actions_emb = []
        for agent_id in idx_state.keys():
            id_obs = idx_state[agent_id].to(self.device)
            # print(f"debug only: @forward: id_obs size: {id_obs.shape}")
            idx_emb = self.idx_emb(id_obs[:, 0].int()).to(self.device)
            idx_obs_emb = torch.concat((idx_emb, id_obs[:, 1:]), dim=1).to(self.device)
            latent_rep = self.encoders[agent_id](idx_obs_emb)
            if self.descrete_act:
                action_emb = self.action_encoder[agent_id](actions[agent_id].int().to(self.device))
            else:
                action_emb = self.action_encoder[agent_id](actions[agent_id].to(self.device))
            mu = latent_rep[:, :self.feature]
            log_var = latent_rep[:, self.feature:]
            z = reparameterize(mu, log_var)
            
            z_all.append(z)
            actions_emb.append(action_emb)
            mu_all.append(mu)
            log_var_all.append(log_var)

        z_all = torch.concat(z_all, dim=-1)
        actions_emb = torch.concat(actions_emb, dim=-1).squeeze()
        print(f"debug only: z size: {z.shape}")
        print(f"debug only: z_all size: {z_all.shape}")
        print(f"debug only: action_emb size: {action_emb.shape}")
        print(f"debug only: actions_emb size: {actions_emb.shape}")
        z_all = torch.concat([z_all, actions_emb], dim=-1)
        # mu_all = torch.concat(mu_all, dim=-1)
        # log_var_all = torch.concat(log_var_all, dim=-1)

        # recon = self.decoder(z_all)
        recon_state = self.state_decoder(z_all)
        recon_reward = self.reward_linear(self.reward_decoder(z_all))
        
        # return recon, mu_all, log_var_all
        return recon_state, recon_reward, mu_all, log_var_all

    def save(self, path):
        torch.save(self.state_dict(), path)


