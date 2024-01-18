from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax
import jax.numpy as jnp
import torch
from transformers import LlamaConfig, LlamaTokenizer

from lib.LLM_params import convert_back_llama
from lib.param_utils import load_params

config = LlamaConfig.from_pretrained('../llama-weights/llama2-7B')
tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/llama2-7B')

cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = load_params('llama2-7B.pickle')
params = jax.tree_map(lambda x: x.astype(jnp.float32), params)  # upcast to float32
model = convert_back_llama(params, config=config)
model.bfloat16()  # cast back to bfloat16
model.cuda()

inputs = tokenizer(['What is the meaning of life?'], return_tensors='pt')
with torch.no_grad():
    generated_seq = model.generate(input_ids=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), do_sample=True, max_new_tokens=30)
decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)
print(decoded_texts)
