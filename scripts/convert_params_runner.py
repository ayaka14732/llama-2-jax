from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import fire
import jax
import jax.numpy as jnp
from transformers import LlamaForCausalLM

from lib.llama import check_llama, model_config_llama1_7B, model_config_llama2_13B, model_config_llama2_70B, model_config_llama2_7B
from lib.llama_params import convert_llama
from lib.param_utils import save_params

pairs = {
    'llama1-7B': ('../llama-weights/llama1-7B', model_config_llama1_7B),
    'llama2-7B': ('meta-llama/Llama-2-7b-hf', model_config_llama2_7B),
    'llama2-13B': ('meta-llama/Llama-2-13b-hf', model_config_llama2_13B),
    'llama2-70B': ('meta-llama/Llama-2-70b-hf', model_config_llama2_70B),
}

def convert(target: str, save_path: str = '') -> None:
    path, model_config = pairs[target]
    model_pt = LlamaForCausalLM.from_pretrained(path)  # cannot use `torch_dtype=torch.bfloat16` here, see https://github.com/pytorch/pytorch/issues/101781
    params = convert_llama(model_pt, model_config=model_config)
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    check_llama(params, model_config=model_config)
    if save_path:
        save_params(params, f'{save_path}/{target}.pickle')
    else:
        save_params(params, f'{target}.pickle')

if __name__ == '__main__':
  fire.Fire(convert)
