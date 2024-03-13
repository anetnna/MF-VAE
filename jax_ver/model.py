import jax
import jax.numpy as jnp
from jax import random, vmap
import optax  # 用于优化
from flax import linen as nn  # 用于定义神经网络模块
import numpy as np
from flax.core import unfreeze

from typing import Dict, Any, Optional


class Encoder(nn.Module):
    in_dim: int
    out_dim: int
    hidden_dims = [64, 64, 256]  # define dimension of hidden layer

    @nn.compact
    def __call__(self, x):
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f"fc{i}")(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x

# 示例使用：
# 需要首先初始化模型和参数
# 例如，如果输入维度是28*28，输出维度设为10
# from flax.core import unfreeze
# from jax import random

# key = random.PRNGKey(0)
# model = Encoder(in_dim=28*28, out_dim=10)
# params = model.init(key, jnp.ones([1, 28*28]))  # 假设输入是28*28维度的
# 输出params是模型的初始参数

class ActionEncoder(nn.Module):
    in_dim: int
    out_dim: int
    hidden_dims = [64]  # 定义隐藏层维度

    @nn.compact
    def __call__(self, x):
        # 输入数据x
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)  # 添加Dense层
            x = nn.relu(x)  # ReLU激活函数
        x = nn.Dense(features=self.out_dim)(x)  # 输出层
        return x

# 示例使用：
# 初始化模型和参数的步骤与之前的例子相同

def reparameterize(mu, log_var, rng_key):
    std = jnp.exp(0.5 * log_var)
    eps = jax.random.normal(rng_key, shape=std.shape)
    sample = mu + eps * std
    return sample

# key = jax.random.PRNGKey(0)  # 创建一个随机数生成器的密钥
# mu = jnp.array([0.0, 0.0])  # 示例均值
# log_var = jnp.array([0.0, 0.0])  # 示例对数方差
# sample = reparameterize(mu, log_var, key)  # 生成样本

class Decoder(nn.Module):
    in_dim: int
    out_dim: int
    hidden_dims = [1024, 256, 64, 256, 1024]  # 定义隐藏层维度

    @nn.compact
    def __call__(self, x):
        # 输入数据x
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)  # 添加Dense层
            x = nn.relu(x)  # ReLU激活函数
        x = nn.Dense(features=self.out_dim)(x)  # 输出层
        return x

# 示例使用：
# 初始化模型和参数的步骤与之前的例子相同


class Embedding(nn.Module):
    num_embeddings: int  # 嵌入的数量，等于词汇表的大小
    embedding_dim: int   # 每个嵌入的维度

    def setup(self):
        # 初始化嵌入矩阵，这里使用正态分布初始化
        # 注意，这里不再需要传递 rng_key，因为 setup 方法在模型初始化时调用
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(),
            (self.num_embeddings, self.embedding_dim)
        )

    def __call__(self, indices):
        # 索引嵌入矩阵以获取对应的嵌入向量
        return self.embedding[indices]


class MAVAE(nn.Module):
    idx_features: int
    obs_features: int
    action_features: int
    descrete_act: bool
    agents: list
    obs_dim: dict
    action_dim: dict
    # 移除device参数，JAX会自动处理设备放置

    def setup(self):
        encoders = {}
        self.idx_emb = Embedding(num_embeddings=len(self.agents), embedding_dim=self.idx_features)
        #encoders = {agent_id: Encoder(self.obs_dim[agent_id] + self.idx_features, 2 * self.obs_features) for agent_id in self.agents}
        action_encoders = {}
        # print(f"debug only: @model call: agents size: {self.agents}")
        for agent_id in self.agents:
            encoders[agent_id] = Encoder(
                self.obs_dim[agent_id] + self.idx_features, 
                2*self.obs_features
            )
            if self.descrete_act:
                action_encoders[agent_id] = Embedding(
                    num_embeddings=self.action_dim[agent_id], 
                    embedding_dim=self.action_features
                )
                # print(f"debug only: action_dim[agent_id]: {self.action_dim[agent_id]}")
            else:
                action_encoders[agent_id] = ActionEncoder(self.action_dim[agent_id], self.action_features)
        self.encoders = encoders
        self.action_encoders = action_encoders

        #decoder_out_dim = len(self.agents) + sum(self.obs_dim.values())
        #decoder = Decoder((self.obs_features + self.action_features) * len(self.agents), decoder_out_dim)
        self.state_decoder = Decoder((self.obs_features + self.action_features) * len(self.agents), sum(self.obs_dim.values()))
        self.reward_decoder = Decoder((self.obs_features + self.action_features) * len(self.agents), len(self.agents))
        self.reward_linear = nn.Dense(features=len(self.agents), use_bias=True)

    @nn.compact
    def __call__(self, 
                 idx_state,
                 actions, 
                 rng_key):
        # 初始化 Embedding 和各种 Encoder
        z_all = []
        mu_all = []
        log_var_all = []
        actions_emb = []

        for agent_id in idx_state.keys():
            id_obs = idx_state[agent_id]
            # print(f"debug only: @model call: {agent_id} id_obs size {id_obs.shape}")
            # 使用 jnp.floor 将浮点数向下取整，然后使用 astype 转换为整型
            idx_emb_output = self.idx_emb(jnp.floor(id_obs[:, 0]).astype(jnp.int32))
            # print(f"debug only: @model call:  {agent_id} idx_emb_output size {idx_emb_output.shape}")
            idx_obs_emb = jnp.concatenate((idx_emb_output, id_obs[:, 1:]), axis=1)
            # print(f"debug only: @model call:  {agent_id} idx_obs_emb size {idx_obs_emb.shape}")
            latent_rep = self.encoders[agent_id](idx_obs_emb)
            # print(f"debug only: @model call:  {agent_id} latent_rep size {latent_rep.shape}")

            # 对每个智能体分割rng_key
            rng_key, sub_key = random.split(rng_key)

            if self.descrete_act:
            # 将动作数据转换为整型，以用作索引
                action_emb = self.action_encoders[agent_id](actions[agent_id].astype(jnp.int32))
                # print(f"debug only: @model call: actions_encoders size: {action_emb.shape}")
            else:
                action_emb = self.action_encoders[agent_id](actions[agent_id])
            
            # print(f"debug only: @model call: {agent_id} action_emb size {action_emb.shape}")
            mu = latent_rep[:, :self.obs_features]
            log_var = latent_rep[:, self.obs_features:]
            
            # 使用分割后的key
            z = reparameterize(mu, log_var, sub_key)
            
            z_all.append(z)
            actions_emb.append(action_emb)
            mu_all.append(mu)
            log_var_all.append(log_var)

        z_all = jnp.concatenate(z_all, axis=1)
        # print(f"debug only: @call: z_all shape: {z_all.shape}")

        actions_emb = jnp.concatenate(actions_emb, axis=1)
        # print(f"debug only: z size: {z.shape}")
        # print(f"debug only: z_all size: {z_all.shape}")
        # print(f"debug only: action_emb size: {action_emb.shape}")
        # print(f"debug only: actions_emb size: {actions_emb.shape}")
        z_all = jnp.concatenate([z_all, actions_emb], axis=1)
        # print(f"debug only: @call: z_all shape: {z_all.shape}")

        recon_state = self.state_decoder(z_all)
        recon_reward = self.reward_linear(self.reward_decoder(z_all))

        mu_all = jnp.concatenate(mu_all, axis=1)
        log_var_all = jnp.concatenate(log_var_all, axis=1)
        
        return recon_state, recon_reward, mu_all, log_var_all

    # JAX/Flax不需要专门的save方法，因为模型的参数可以直接通过外部函数保存