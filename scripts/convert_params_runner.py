from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from transformers import LlamaForCausalLM

from lib.model import check_llama, config_7B
from lib.param_utils import convert_llama, save_params

model_pt = LlamaForCausalLM.from_pretrained('../llama-weights/converted/7B')
params = convert_llama(model_pt, config=config_7B)
check_llama(params, config=config_7B)
save_params(params, '7B.pickle')
