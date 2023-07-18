import json
import os
from torch.utils.data import Dataset
from typing import Literal, Union

def load_data(*, split=Union[Literal['train'], Literal['test']]):
    path = os.path.join(f'../grade-school-math/grade_school_math/data/{split}.jsonl')
    res = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            question = 'Question:\n' + data['question'] + '\n'
            answer = 'Answer:\n' + data['answer']
            answer = answer.replace('#### ', 'Final answer:\n')
            res.append((question, answer))
    return res

class GSMDataset(Dataset):
    def __init__(self, *, split=Union[Literal['train'], Literal['test']]) -> None:
        self.data = load_data(split=split)
        super().__init__()

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)
