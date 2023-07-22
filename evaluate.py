# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_gpu; initialise_gpu(cuda_visible_devices='0,1')

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
from lib.model import model_config_llama2_7B
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER, HASHED_BUDDHA

def main() -> None:
    jax_smi.initialise_tracking(interval=0.5)

    key = rand.PRNGKey(BEST_INTEGER)
    params = load_params('woven-oath-70.pickle')
    max_len = 640
    batch_size = 2
    seed = HASHED_BUDDHA

    from lib.model.llama import shard_llama
    from lib.tree_utils import stack_leaves
    params = params._replace(model=params.model._replace(decoder=stack_leaves(params.model.decoder)))
    params = shard_llama(params)
    from jax.sharding import PositionalSharding; devices = jax.devices(); shards = PositionalSharding(devices); n_shard = len(devices)
    shard_all = lambda x: jax.tree_map(lambda i: jax.device_put(i, shards.replicate((0,))), x)

    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=640, top_p=0.9)
    dataset = GSMDataset(split='test')
    collate_fn = partial(gsm_collate_fn_test, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)
    config_llama2_7B_ = model_config_llama2_7B._replace(dropout_rate=None)

    pbar = tqdm(total=len(dataloader))
    with open('results.txt', 'w', encoding='utf-8') as f:
        for seq, seq_mask, labels in dataloader:
            key, subkey = rand.split(key)

            seq, seq_mask = shard_all((seq, seq_mask))

            generated_seq = top_p(params, seq, seq_mask, key=subkey, model_config=config_llama2_7B_, top_p_config=top_p_config)
            decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

            for decoded_text, label in zip(decoded_texts, labels):
                print(json.dumps([decoded_text, label], ensure_ascii=False), file=f)
                f.flush()
            pbar.update()

if __name__ == '__main__':
    main()
