# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)

import jax.numpy as jnp
from jax_smi import initialise_tracking
from transformers import LlamaTokenizer

from lib.model import llama_model, config_7B
from lib.param_utils import load_params

initialise_tracking()

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights')
params = load_params('7B.pickle')

inputs = tokenizer(["J'aime le chocolat = I like chocolate\n祝你一天过得愉快 ="], return_tensors='jax')
seq = inputs.input_ids.astype(jnp.uint16)
attn_mask = inputs.attention_mask.astype(jnp.bool_)
outputs = llama_model(params.model, seq, attn_mask, config=config_7B)
logits = outputs[:, -1] @ params.lm_head
max_logit = logits.argmax(-1)
tokenizer.decode(max_logit)
