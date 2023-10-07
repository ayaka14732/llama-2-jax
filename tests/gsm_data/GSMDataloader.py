from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from functools import partial

from transformers import LlamaTokenizer

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, gsm_collate_fn_train

batch_size = 1
seed = 0
max_len = 512

def main() -> None:
    dataset = GSMDataset(split='train')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    for i, (seq, seq_mask, labels, labels_mask) in enumerate(dataloader):
        print('seq', seq)
        print('seq_mask', seq_mask)
        print('labels', labels)
        print('labels_mask', labels_mask)

        if i == 2:
            break

if __name__ == '__main__':
    main()
