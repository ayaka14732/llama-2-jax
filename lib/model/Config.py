from typing import NamedTuple

class Config(NamedTuple):
    d_k: int
    d_model: int
    d_v: int
    dropout_rate: int
    rms_norm_eps: float
    n_heads: int
    n_layers: int
    vocab_size: int

config_7B = Config(
    d_k=128,
    d_model=4096,
    d_v=128,
    dropout_rate=0.1,
    rms_norm_eps=1e-6,
    n_heads=32,
    n_layers=32,
    vocab_size=32000,
)
