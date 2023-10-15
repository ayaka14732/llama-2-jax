import jax
import jax.random as rand
import jax_smi
import json
from transformers import LlamaTokenizer
from functools import partial
from tqdm import tqdm

from lib.dataloader import LlamaDataLoader
from lib.generation import TopPGenerationConfig, top_p
from lib.gsm_data import GSMDataset, gsm_collate_fn_test
from lib.llama import model_config_llama2_7B
from lib.multihost_utils import shard_model_params
from lib.param_utils import load_params
from lib.proc_init_utils import initialise_tpu
from lib.seeding import BEST_INTEGER, HASHED_BUDDHA

def main() -> None:
    initialise_tpu('v4-16', n_devices=8, rank=0)
    is_process_0 = jax.process_index() == 0
    if is_process_0:
        jax_smi.initialise_tracking(interval=0.5)

    key = rand.key(BEST_INTEGER, impl='rbg')
    max_len = 640
    batch_size = 2
    seed = HASHED_BUDDHA

    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params('glowing-terrain-95.pickle')
    params = shard_model_params(params)
    if is_process_0:
        print('Successfully loaded and sharded model parameters!')

    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=640, top_p=0.9)
    dataset = GSMDataset(split='test')
    collate_fn = partial(gsm_collate_fn_test, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)
    config_llama2_7B_ = model_config_llama2_7B._replace(dropout_rate=None)

    if is_process_0:
        pbar = tqdm(total=len(dataloader))
    with open('results.txt', 'w', encoding='utf-8') as f:
        for seq, seq_mask, labels in dataloader:
            key, subkey = rand.split(key)

            generated_seq = top_p(params, seq, seq_mask, key=subkey, model_config=config_llama2_7B_, top_p_config=top_p_config)
            decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

            if is_process_0:
                for decoded_text, label in zip(decoded_texts, labels):
                    print(json.dumps([decoded_text, label], ensure_ascii=False), file=f)
                    f.flush()
                pbar.update()

if __name__ == '__main__':
    main()
