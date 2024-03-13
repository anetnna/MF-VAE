import jax
import jax.numpy as jnp
from jaxmarl import make
import flashbax as fbx 

from typing import Dict, Any, Optional

def create_joint_transition(obs: Dict[str, Any], 
                            reward: Dict[str, float], 
                            action: Dict[str, Any], 
                            next_obs: Dict[str, Any],
                            done: Dict[str, bool]) -> Optional[Dict[str, Any]]:
    """
    Create a joint transition dictionary from individual observations, rewards, actions, done, and next observations of agents.

    Args:
        obs (Dict[str, Any]): A dictionary mapping each agent's ID to its current observation.
        reward (Dict[str, float]): A dictionary mapping each agent's ID to its received reward.
        action (Dict[str, Any]): A dictionary mapping each agent's ID to its action.
        next_obs (Dict[str, Any]): A dictionary mapping each agent's ID to its next step observation.
        done (Dict[str, bool]): A dictionary mapping each agent's ID to its done status (whether the episode is over).

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
    ma_done = 0.0
    for agent_id in agents:
        # Check if all required keys are present for the agent
        if agent_id not in reward or agent_id not in action or agent_id not in next_obs or agent_id not in done:
            print(f"agent id {agent_id} not exist in action/reward/next_obs/done dict")
            return None

        # Add each component of the transition to the joint transition map
        joint_transition_map[f"{agent_id}_obs"] = obs[agent_id].reshape((-1, 1))
        joint_transition_map[f"{agent_id}_act"] = action[agent_id].reshape((-1, 1))
        joint_transition_map[f"{agent_id}_next_obs"] = next_obs[agent_id].reshape((-1, 1))
        # print(f"debug only: reward shape {obs[agent_id].shape}")
        joint_transition_map[f"{agent_id}_rew"] = reward[agent_id].reshape((-1, 1))
        # joint_transition_map[f"{agent_id}_done"] = done[agent_id]
        if done[agent_id]:
            ma_done = 1.0
    device_info = obs[agent_id].device()
    joint_transition_map["done"] = jax.device_put(jnp.array(ma_done).reshape((-1, 1)), device_info)

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
                                                 done)
        dummy_transiton_map = generate_dummy_transition(transition_map)
        self.buffer_state = self.buffer.init(dummy_transiton_map)
        print_transition_shape(dummy_transiton_map)
    
    def add_trans(self, 
                  obs: Dict[str, Any], 
                  reward: Dict[str, float], 
                  actions: Dict[str, Any],  
                  next_obs: Dict[str, Any],
                  done: Dict[str, bool]):
        if self.buffer_state is None:
            print(f"buffer not init; please call init_buffer() first")
            return
        transition_map = create_joint_transition(obs, 
                                                 reward, 
                                                 actions, 
                                                 next_obs, 
                                                 done)
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


if __name__ == "__main__":
    devices = jax.devices()
    print("Available devices:", devices)

    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step, key_sample = jax.random.split(key, 5)

    # Initialise environment.
    env = make('MPE_simple_world_comm_v3')

    # Reset the environment.
    obs, state = env.reset(key_reset)

    # Sample random actions.
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_act[i]) \
        for i, agent in enumerate(env.agents)}

    # Perform the step transition.
    next_obs, state, reward, done, infos = env.step(key_step, state, actions)

    joint_transition_map = create_joint_transition(obs, 
                                                   reward, 
                                                   actions, 
                                                   next_obs, 
                                                   done)
    buffer = JaxFbxBuffer(max_length=80, 
                          min_length=64, 
                          batch_size=64, 
                          add_batch=False)
    buffer.init_buffer(obs, reward, actions, next_obs, done)
    # buffer.add_trans(obs, reward, actions, next_obs, done)
    obs = next_obs

    for _ in range(100):
        # key_act = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_act[i]) \
            for i, agent in enumerate(env.agents)}
        next_obs, state, reward, done, infos = env.step(key_step, state, actions)
        buffer.add_trans(obs, reward, actions, next_obs, done)
        obs = next_obs

    batch = buffer.sample(key_sample)
    # batch.experience: {k: v}
    # value shape: (batch_size, value_shape[0], value_shape[1], ...)
    print(batch.experience['adversary_1_next_obs'].shape)
