from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

import os

from typing import Dict, Union

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
        flattened_dict = flatten_dict(params, sep=',')
        save_file(flattened_dict, filename)

def load_params(filename):
    return load_file(filename)

def reconstruct_params(flattened_params):
    return unflatten_dict(flattened_params, sep=',')