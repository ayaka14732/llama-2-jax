import multiprocessing
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _collate_fn_t

from ..proc_init_utils import initialise_cpu

def worker_init_fn(worker_id: int) -> None:
    initialise_cpu()  # process data on CPU
    worker_seed = torch.initial_seed() % 2**32  # TODO: why?
    random.seed(worker_seed)

class LlamaDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, collate_fn: _collate_fn_t, batch_size: int, seed: int) -> None:
        gen = torch.Generator()
        gen.manual_seed(seed)  # TODO: key?
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
            # worker_init_fn=worker_init_fn,
            # prefetch_factor=1,
            # multiprocessing_context=multiprocessing.get_context('spawn'),
            generator=gen,
            # persistent_workers=True,
        )
