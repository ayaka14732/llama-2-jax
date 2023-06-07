from functools import partial
import jax
from jax import Array
from typing import NamedTuple

from .Config import Config
from .decoder import Decoder

class Llama(NamedTuple):
    embedding: Array
    decoder: Decoder
    norm: Array

@partial(jax.jit, static_argnames=('config',))
def llama(params: Llama, *, config=Config):
    pass
