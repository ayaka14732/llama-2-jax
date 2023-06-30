# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from transformers import LlamaTokenizer

from lib.model import llama, config_7B
from lib.param_utils import load_params

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights')
params = load_params('7B.pickle')

inputs = tokenizer(["J'aime le chocolat = I like chocolate\n祝你一天过得愉快 ="], return_tensor='jax')
outputs = llama(params, inputs.input_ids, inputs.attention_mask, config=config_7B)
