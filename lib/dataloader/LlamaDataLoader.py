import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _collate_fn_t

class LlamaDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, collate_fn: _collate_fn_t, batch_size: int, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=gen,
        )
