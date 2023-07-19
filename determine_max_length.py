from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from transformers import LlamaTokenizer

from lib.gsm_data.GSMDataset import load_data

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/7B')
data = load_data(split='train')

len_list = []

for question, answer in data:
    tokenized_ids = tokenizer(question + answer).input_ids
    len_list.append(len(tokenized_ids))

print(max(len_list))

cutoff = 512
n_total = len(data)
n_cut = sum(length > cutoff for length in len_list)
print(f'n_cut: {n_cut}, n_total {n_total}')
