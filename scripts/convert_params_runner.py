from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import fire
from transformers import LlamaForCausalLM

from lib.model import check_llama, config_7B, config_llama2_7B
from lib.param_utils import convert_llama, save_params

pairs = {
    '7B': ('../llama-weights/7B', config_7B),
    'llama2-7B': ('../llama-weights/llama2-7B', config_llama2_7B),
    # 'llama2-70B': ('../llama-weights/llama2-70B', config_70B),
}

def convert(target: str) -> None:
    path, model_config = pairs[target]
    model_pt = LlamaForCausalLM.from_pretrained(path)
    params = convert_llama(model_pt, model_config=model_config)
    check_llama(params, model_config=model_config)
    save_params(params, f'{target}.pickle')

if __name__ == '__main__':
  fire.Fire(convert)
