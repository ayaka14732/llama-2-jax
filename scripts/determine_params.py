from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

q_size = model.model.layers[0].self_attn.q_proj.weight.shape[0] * model.model.layers[0].self_attn.q_proj.weight.shape[1]
k_size = model.model.layers[0].self_attn.k_proj.weight.shape[0] * model.model.layers[0].self_attn.k_proj.weight.shape[1]

c = model.model.embed_tokens.weight.shape[0]
f = model.model.layers[0].mlp.gate_proj.weight.shape[0]
m = model.model.layers[0].mlp.gate_proj.weight.shape[1]
n = len(model.model.layers)
r = q_size // k_size
h = model.config.num_attention_heads // r
k = k_size // m // h

n_params = sum(x.numel() for x in model.parameters())

print('C', c)
print('N', n)
print('K', k)
print('H', h)
print('R', r)
print('M', m)
print('F', f)
print('n_params', n_params)
