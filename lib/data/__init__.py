from typing import NamedTuple

from jax import Array

class TrainData(NamedTuple):
    seq: Array
    seq_mask: Array
    labels: Array
    labels_mask: Array

class TestData(NamedTuple):
    seq: Array
    seq_mask: Array
    labels: list[str]
