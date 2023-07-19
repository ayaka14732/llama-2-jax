import jax
import jax.numpy as jnp

from lib.param_utils import load_params, save_params

params = load_params('llama2-7B.pickle')
params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
save_params(params, 'llama2-7B-float16.pickle')
