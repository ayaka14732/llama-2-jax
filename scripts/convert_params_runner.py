from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu
initialise_cpu()
from huggingface_hub import HfApi
import torch
import fire
import jax
from transformers import LlamaForCausalLM
import gc
from lib.llama import check_llama, model_config_llama1_7B, model_config_llama2_13B, model_config_llama2_70B, model_config_llama2_7B
from lib.llama_params import convert_llama
from lib.param_utils import save_params
import aqt.jax.v2.flax.aqt_flax as aqt
import aqt.jax.v2.config as aqt_config
import jax.numpy as jnp

pairs = {
    'llama1-7B': ('../llama-weights/llama1-7B', model_config_llama1_7B),
    'llama2-7B': ('meta-llama/Llama-2-7b-hf', model_config_llama2_7B),
    'llama2-13B': ('meta-llama/Llama-2-13b-hf', model_config_llama2_13B),
    'llama2-70B': ('meta-llama/Llama-2-70b-hf', model_config_llama2_70B),
    'llama2-7B-chat': ('meta-llama/Llama-2-7b-chat-hf', model_config_llama2_7B),
    'llama2-13B-chat': ('meta-llama/Llama-2-13b-chat-hf', model_config_llama2_13B),
    'llama2-70B-chat': ('meta-llama/Llama-2-70b-chat-hf', model_config_llama2_70B),
}

def convert(target: str, save_path: str = '') -> None:
    path, model_config = pairs[target]
    model_pt = LlamaForCausalLM.from_pretrained(path, torch_dtype= torch.float16)
    params = convert_llama(model_pt, model_config=model_config)
    del model_pt
    
    gc.collect()
    # Quantize the parameters to 8-bit using AQT
    aqt_config = aqt.AqtConfig(forward_dtype=jnp.int8, backward_dtype=jnp.int8)
    params = aqt.inject(params, aqt_config)
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
