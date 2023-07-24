from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import matplotlib.pyplot as plt
from transformers import LlamaTokenizer

from lib.gsm_data.GSMDataset import load_data

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/7B')
data = load_data(split='train')

len_list = []

for question, answer in data:
    tokenized_ids = tokenizer(question + answer).input_ids
    len_list.append(len(tokenized_ids))

print('Max length:', max(len_list))

max_val = max(len_list) + 25

plt.figure(figsize=(10, 6))
plt.hist(len_list, bins=range(0, max_val, 25), edgecolor='black')
plt.title('GSM Data Length Distribution')
plt.xlabel('Ranges (Every 25)')
plt.ylabel('Count')
plt.xticks(range(0, max_val, 25))
plt.grid(True)

plt.tight_layout()
plt.savefig('len_dist.png')
