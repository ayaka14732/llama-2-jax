from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()
from huggingface_hub import HfApi

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
    'llama2-70B-8bit':('caisarl76/llama2-70B-8bit', model_config_llama2_70B),
}

def convert(target: str, save_path: str = '') -> None:
    path, model_config = pairs[target]
    model_pt = LlamaForCausalLM.from_pretrained(path)
    params = convert_llama(model_pt, model_config=model_config)
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    check_llama(params, model_config=model_config)

    print(f'Converted parameters for {target}')
    # Define the file name and path
    file_name = f'{target}.pickle'
    if save_path:
        file_path = f'{save_path}/{file_name}'
    else:
        file_path = file_name

    # Save the parameters
    save_params(params, file_path)
    print(f'Parameters saved to {file_path}')
    # Upload the file to Hugging Face
    print('Uploading to Hugging Face...')
    api = HfApi()
    repo_id = f'divyapatel4/{target}-hf-jax'  # Define the repo_id based on the target
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == '__main__':
  fire.Fire(convert)

