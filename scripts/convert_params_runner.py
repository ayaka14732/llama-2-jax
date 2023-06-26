from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from transformers import LlamaModel

from lib.model import config_7B
from lib.param_utils import convert_llama, save_params

model_pt = LlamaModel.from_pretrained('../llama-weights/converted/7B')
model_jax = convert_llama(model_pt, config=config_7B)
save_params(model_jax, '7B.pickle')
