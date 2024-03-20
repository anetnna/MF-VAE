import jax
import jax.numpy as jnp
from jaxmarl import make
import flashbax as fbx 

from typing import Dict, Any, Optional, Union, NamedTuple

def create_joint_transition(obs: Dict[str, Any], 
                            reward: Dict[str, float], 
                            action: Dict[str, Any], 
                            next_obs: Union[Dict[str, Any], None],
                            done: Dict[str, bool],
                            batch_input: bool=False,
                            traj_input: bool=False) -> Optional[Dict[str, Any]]:
    """
    Create a joint transition dictionary from individual observations, rewards, actions, done, and next observations of agents.

    Args:
        obs (Dict[str, Any]): A dictionary mapping each agent's ID to its current observation.
        reward (Dict[str, float]): A dictionary mapping each agent's ID to its received reward.
        action (Dict[str, Any]): A dictionary mapping each agent's ID to its action.
        next_obs (Dict[str, Any]): A dictionary mapping each agent's ID to its next step observation.
        done (Dict[str, bool]): A dictionary mapping each agent's ID to its done status (whether the episode is over).
        batch_input (bool): A bool indicating whether input is in batch ot not. If yes, the shape would be: [batch_size, ...]
        traj_input (bool): A bool indicating whether input is a trajectory. If yes, the shape would be: [traj_len, ...]

        if batch_input and traj_input: input size is [traj_len, batch_size, ...]
        if not batch_input and traj_input: input size is [traj_len, ...]
        if batch_input and not traj_input: input size is [batch_size, ...]

    Returns:
        Optional[Dict[str, Any]]: A dictionary that contains the joint transition information for all agents, or None if
                                  there is a missing key in reward, action, or next_obs for any agent.

    This function constructs a joint transition map where each agent's observation, action, reward, next observation, and
    done status are stored. The keys are in the format of '<agent_id>_<info>' (e.g., 'agent1_obs', 'agent1_act').
    """
    # Retrieve all agent IDs from observations
    agents = obs.keys()

    # Initialize dictionary to store joint transitions
    joint_transition_map = {}

    # Iterate over each agent to construct their individual transition
    ma_done = None
    for agent_id in agents:
        # Check if all required keys are present for the agent
        if next_obs is not None:
            if agent_id not in reward or agent_id not in action or agent_id not in next_obs or agent_id not in done:
                print(f"agent id {agent_id} not exist in action/reward/next_obs/done dict")
                return None
        else:
            if agent_id not in reward or agent_id not in action or agent_id not in done:
                print(f"agent id {agent_id} not exist in action/reward/done dict")
                return None
        # Add each component of the transition to the joint transition map
        if not batch_input and not traj_input:
            joint_transition_map[f"{agent_id}_obs"] = obs[agent_id].reshape((-1, 1))
            joint_transition_map[f"{agent_id}_act"] = action[agent_id].reshape((-1, 1)).astype(jnp.float32)
            if next_obs is not None:
                joint_transition_map[f"{agent_id}_next_obs"] = next_obs[agent_id].reshape((-1, 1))
            joint_transition_map[f"{agent_id}_rew"] = reward[agent_id].reshape((-1, 1)).astype(jnp.float32)
            if done[agent_id]:
                device_info = obs[agent_id].device()
                ma_done = jax.device_put(jnp.array(True).reshape((-1, 1)), device_info).astype(jnp.float32)
        elif batch_input and not traj_input:
            joint_transition_map[f"{agent_id}_obs"] = obs[agent_id][:, :, jnp.newaxis]
            joint_transition_map[f"{agent_id}_act"] = action[agent_id][:, jnp.newaxis, jnp.newaxis].astype(jnp.float32)
            if next_obs is not None:
                joint_transition_map[f"{agent_id}_next_obs"] = next_obs[agent_id][:, :, jnp.newaxis]
            joint_transition_map[f"{agent_id}_rew"] = reward[agent_id][:, jnp.newaxis, jnp.newaxis].astype(jnp.float32)
            ma_done = done['__all__'][:, jnp.newaxis, jnp.newaxis].astype(jnp.float32)
        elif batch_input and traj_input:
            joint_transition_map[f"{agent_id}_obs"] = obs[agent_id]
            joint_transition_map[f"{agent_id}_act"] = action[agent_id].astype(jnp.float32)
            if next_obs is not None:
                joint_transition_map[f"{agent_id}_next_obs"] = next_obs[agent_id]
            joint_transition_map[f"{agent_id}_rew"] = reward[agent_id].astype(jnp.float32)
            ma_done = done['__all__'].astype(jnp.float32)
        else:
            joint_transition_map[f"{agent_id}_obs"] = obs[agent_id]
            joint_transition_map[f"{agent_id}_act"] = action[agent_id].astype(jnp.float32)
            if next_obs is not None:
                joint_transition_map[f"{agent_id}_next_obs"] = next_obs[agent_id]
            joint_transition_map[f"{agent_id}_rew"] = reward[agent_id].astype(jnp.float32)
            ma_done = done['__all__'].astype(jnp.float32)
    if ma_done is None:
        device_info = obs[agent_id].device()
        ma_done = jax.device_put(jnp.array(False).reshape((-1, 1)), device_info).astype(jnp.float32)
    joint_transition_map["done"] = ma_done
    # print(f"debug only: rew {ma_done.shape}")
    return joint_transition_map

def print_transition_shape(transition: Dict[str, jnp.ndarray]):
     for k, v in transition.items():
        print(f"key {k} with shape: {v.shape} and type {v.dtype}")

def generate_dummy_transition(transition: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
      """
      Generate a dummy transition (filled with zeros) based on the given transition map.

      Args:
          transition (Dict[str, jnp.ndarray]): A dictionary representing a transition in 
                                               the environment. This dictionary should 
                                               contain keys like 'agent_obs', 'agent_act', 
                                               etc., with their corresponding JAX arrays.

      Returns:
          Dict[str, jnp.ndarray]: A dictionary with the same keys as the input transition map, 
                                  where each value is a dummy JAX array of zeros with the same 
                                  shape as the corresponding value in the input transition map.
      """
      dummy_transition = {k: jnp.zeros_like(v) for k, v in transition.items()}
      return dummy_transition

class JaxFbxBuffer:
    def __init__(self, 
                 max_length: int=50_000, 
                 min_length: int=64, 
                 batch_size: int=64,
                 add_batch: bool=False):
        self.buffer = fbx.make_item_buffer(
            max_length=max_length, 
            min_length=min_length, 
            sample_batch_size=batch_size, 
            add_batches=add_batch,
            )
        self.buffer_state = None
    
    def init_buffer(self, 
                    obs: Dict[str, Any], 
                    reward: Dict[str, float], 
                    actions: Dict[str, Any],
                    next_obs: Dict[str, Any], 
                    done: Dict[str, bool]):
        transition_map = create_joint_transition(obs, 
                                                 reward, 
                                                 actions, 
                                                 next_obs, 
                                                 done,
                                                 batch_input=False)
        dummy_transiton_map = generate_dummy_transition(transition_map)
        self.buffer_state = self.buffer.init(dummy_transiton_map)
        print_transition_shape(dummy_transiton_map)
    
    def add_trans(self, 
                  obs: Dict[str, Any], 
                  reward: Dict[str, float], 
                  actions: Dict[str, Any],  
                  next_obs: Dict[str, Any],
                  done: Dict[str, bool],
                  batch_input: bool=False):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        transition_map = create_joint_transition(obs, 
                                                 reward, 
                                                 actions, 
                                                 next_obs, 
                                                 done,
                                                 batch_input)
        # print(transition_map)
        self.buffer_state = self.buffer.add(self.buffer_state, transition_map)
    
    def can_sample(self):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        return self.buffer.can_sample(self.buffer_state)
    
    def sample(self, rng_key):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        if not self.can_sample():
            print(f"can not sample now")
            return
        batch = self.buffer.sample(self.buffer_state, rng_key)
        return batch


class JaxFbxTrajBuffer:
    def __init__(self, 
                 max_length_time_axis: int=50_000, 
                 min_length_time_axis: int=64, 
                 sample_batch_size: int=64,
                 add_batch_size: int=8, # for samplified, add_batch_size=env_num
                 sample_sequence_length: int=1,
                 period: int=1):
        self.buffer = fbx.make_trajectory_buffer(max_length_time_axis=max_length_time_axis,
                                                 min_length_time_axis=min_length_time_axis,
                                                 sample_batch_size=sample_batch_size,
                                                 add_batch_size=add_batch_size,
                                                 sample_sequence_length=sample_sequence_length,
                                                 period=period)
        self.buffer_state = None
    
    def init_buffer(self, 
                    obs: Dict[str, Any], 
                    reward: Dict[str, float], 
                    actions: Dict[str, Any], 
                    done: Dict[str, bool]):
        transition_map = create_joint_transition(obs=obs, 
                                                 reward=reward, 
                                                 action=actions, 
                                                 next_obs=None, 
                                                 done=done,
                                                 batch_input=False,
                                                 traj_input=True)
        dummy_transiton_map = generate_dummy_transition(transition_map)
        self.buffer_state = self.buffer.init(dummy_transiton_map)
    
    def add_trans(self, 
                  obs: Dict[str, Any], 
                  reward: Dict[str, float], 
                  actions: Dict[str, Any], 
                  done: Dict[str, bool],
                  batch_input: bool,):
        transition_map = create_joint_transition(obs=obs, 
                                                 reward=reward, 
                                                 action=actions, 
                                                 next_obs=None, 
                                                 done=done,
                                                 batch_input=batch_input,
                                                 traj_input=True)
        buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, jnp.newaxis], # put the batch dim first and add a dummy sequence dim
                transition_map
            ) # (num_envs, 1, time_steps, ...)
        self.buffer_state = self.buffer.add(self.buffer_state, buffer_traj_batch)
    
    def can_sample(self):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        return self.buffer.can_sample(self.buffer_state)
    
    def sample(self, rng_key):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        if not self.can_sample():
            print(f"can not sample now")
            return
        batch = self.buffer.sample(self.buffer_state, rng_key).experience
        batch = jax.tree_map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                batch
            ) # (max_time_steps, batch_size, ...)
        return batch


class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict


def env_sample_step(env_state, unused):
    # Sample random actions.
    actions = {agent: env.batch_sample(key_act_agents[i], agent) \
        for i, agent in enumerate(env.agents)}

    # Perform the step transition.
    obs, env_state, reward, done, infos = env.batch_step(key_step, env_state, actions)
    return env_state, Transition(obs, actions, reward, done)


if __name__ == "__main__":
    devices = jax.devices()
    print("Available devices:", devices)

    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step, key_sample = jax.random.split(key, 5)

    # Initialise environment.
    from env import EnvRolloutManager

    env_batch_size = 8
    plain_env = make('MPE_simple_tag_v3')
    env = EnvRolloutManager(plain_env, batch_size=env_batch_size)

    # Reset the environment.
    _, state = env.batch_reset(key_reset)

    # # Sample random actions.
    key_act_agents = jax.random.split(key_act, env.num_agents)
    # actions = {agent: env.action_space(agent).sample(key_act[i]) \
    #     for i, agent in enumerate(env.agents)}

    # # Perform the step transition.
    # next_obs, state, reward, done, infos = env.step(key_step, state, actions)
    # _, obs, actions, reward, done = env_sample_step(state)

    _, trans = jax.lax.scan(
        env_sample_step, state, None, 10
    )
    trans_unbatched = jax.tree_map(lambda x: x[:, 0], trans) # remove the NUM_ENV dim
    obs, actions, reward, done = trans_unbatched.obs, trans_unbatched.actions, trans_unbatched.rewards, trans_unbatched.dones
    
    # joint_transition_map = create_joint_transition(obs, 
    #                                                reward, 
    #                                                actions, 
    #                                                next_obs, 
    #                                                done)
    # buffer = JaxFbxBuffer(max_length=80, 
    #                       min_length=64, 
    #                       batch_size=64, 
    #                       add_batch=False)
    buffer = JaxFbxTrajBuffer(max_length_time_axis=500, 
                 min_length_time_axis=64, 
                 sample_batch_size=8,
                 add_batch_size=env_batch_size, # for samplified, add_batch_size=
                 sample_sequence_length=1,
                 period=1)
    buffer.init_buffer(obs, reward, actions, done)
    # buffer.add_trans(obs, reward, actions, next_obs, done)
    # obs = next_obs
    for _ in range(100):
        _, state = env.batch_reset(key_reset)
        _, trans = jax.lax.scan(
            env_sample_step, state, None, 10
        )
        # buffer_traj_batch = jax.tree_util.tree_map(
        #     lambda x:jnp.swapaxes(x, 0, 1)[:, jnp.newaxis], # put the batch dim first and add a dummy sequence dim
        #     trans
        # ) # (num_envs, 1, time_steps, ...)
        obs, actions, reward, done = trans.obs, trans.actions, trans.rewards, trans.dones
        buffer.add_trans(obs, actions, reward, done, batch_input=True)


    # for _ in range(100):
    #     # key_act = jax.random.split(key_act, env.num_agents)
    #     actions = {agent: env.action_space(agent).sample(key_act[i]) \
    #         for i, agent in enumerate(env.agents)}
    #     next_obs, state, reward, done, infos = env.step(key_step, state, actions)
    #     buffer.add_trans(obs, reward, actions, next_obs, done)
    #     obs = next_obs


    batch = buffer.sample(key_sample)
    # batch.experience: {k: v}
    # value shape: (batch_size, value_shape[0], value_shape[1], ...)
    print(batch['adversary_1_obs'].shape)
