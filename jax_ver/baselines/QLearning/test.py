"""
Test between different algorithms
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import NamedTuple, Dict, Union, Any
from collections import defaultdict

import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_dir)

from src.env import get_space_dim, EnvRolloutManager
from jaxmarl import make
from src.jax_buffer import JaxFbxTrajBuffer, Transition
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
