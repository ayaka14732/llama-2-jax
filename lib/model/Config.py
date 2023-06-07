from typing import TypedDict

class Config(TypedDict):
    d_k: int
    d_model: int
    d_v: int
    dropout_rate: int
    n_heads: int
    n_layers: int
    vocab_size: int

_config: Config = {
    'd_k': 128,
    'd_model': 4096,
    'd_v': 128,
    'dropout_rate': 0.1,
    'n_heads': 32,
    'n_layers': 32,
    'vocab_size': 32000,
}
