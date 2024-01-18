from lib.proc_init_utils import initialise_gpu; initialise_gpu(cuda_visible_devices='1')

from functools import partial
from itertools import chain, repeat
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm
from transformers import LlamaConfig, LlamaTokenizer

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset
from lib.LLM_params import convert_back_llama
from lib.param_utils import load_params
from lib.seeding import HASHED_BUDDHA

class TestData(NamedTuple):
    seq: torch.Tensor
    seq_mask: torch.Tensor
    labels: list[str]

def gsm_collate_fn_test(tokenizer: LlamaTokenizer, max_len: int, data_batch: list[tuple[str, str]]):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    seq_list = []
    seq_mask_list = []
    labels_list = []

    for question, answer in data_batch:
        question = tokenizer(question, add_special_tokens=False, return_attention_mask=False).input_ids

        len_question = len(question)
        len_seq = len_question + 1
        len_pad = max_len - len_seq

        assert len(question) + 1 < max_len, '`max_len` too small'

        seq = list(chain((bos_id,), question, repeat(eos_id, len_pad)))
        seq_mask = list(chain(repeat(True, 1 + len_question), repeat(False, len_pad)))

        seq = seq[:max_len]
        seq_mask = seq_mask[:max_len]

        seq_list.append(seq)
        seq_mask_list.append(seq_mask)
        labels_list.append(answer)

    seq_ = torch.tensor(seq_list, dtype=torch.int32, device='cuda')
    seq_mask_ = torch.tensor(seq_mask_list, dtype=torch.bool, device='cuda')

    return TestData(seq_, seq_mask_, labels_list)


config = LlamaConfig.from_pretrained('../llama-weights/llama2-7B')
tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/llama2-7B')

cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = load_params('llama2-7B.pickle')
    params = jax.tree_map(lambda x: x.astype(jnp.float32), params)  # upcast to float32
model = convert_back_llama(params, config=config)
model.bfloat16()  # cast back to bfloat16
model.cuda()


max_len = 640
batch_size = 2
seed = HASHED_BUDDHA

dataset = GSMDataset(split='test')
collate_fn = partial(gsm_collate_fn_test, tokenizer, max_len)
dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

with open('results-zero-shot.txt', 'w', encoding='utf-8') as f:
    for seq, seq_mask, labels in tqdm(dataloader):
        with torch.no_grad():
            generated_seq = model.generate(input_ids=seq, attention_mask=seq_mask, do_sample=True, max_new_tokens=640)
        decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)
        for decoded_text, label in zip(decoded_texts, labels):
            print(json.dumps(decoded_text), file=f)
            print(json.dumps(label), file=f)
            f.flush()
