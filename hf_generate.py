from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax
import jax.numpy as jnp
from transformers import LlamaConfig, LlamaTokenizer

from lib.param_utils import convert_back_llama, load_params

cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = load_params('llama2-7B.pickle')

config = LlamaConfig.from_pretrained('/dev/shm/llama-weights/llama2-7B')
tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/llama2-7B')
params = jax.tree_map(lambda x: x.astype(jnp.float32), params)  # upcast to float32
model = convert_back_llama(params, config=config)
model.bfloat16()

inputs = tokenizer(['What is the meaning of life?'], return_tensors='pt')
model.generate(**inputs)
