from typing import NamedTuple, Optional

class ModelConfig(NamedTuple):
    d_ff: int
    d_k: int
    d_model: int
    d_v: int
    dropout_rate: Optional[float]
    n_heads_kv: int
    n_layers: int
    n_rep_kv: int
    rms_norm_eps: float
    token_id_bos: int
    token_id_eos: int
    token_id_pad: int
    vocab_size: int

model_config_dummy = ModelConfig(
    d_ff=17,
    d_k=32,
    d_model=15,
    d_v=16,
    dropout_rate=0.1,
    n_heads_kv=32,
    n_layers=3,
    n_rep_kv=2,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=5,
)

model_config_llama1_7B = ModelConfig(
    d_ff=11008,
    d_k=128,
    d_model=4096,
    d_v=128,
    dropout_rate=0.1,
    n_heads_kv=32,
    n_layers=32,
    n_rep_kv=1,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
)

model_config_llama2_7B = model_config_llama1_7B

model_config_llama2_70B = ModelConfig(
    d_ff=28672,
    d_k=128,
    d_model=8192,
    d_v=128,
    dropout_rate=0.1,
    n_heads_kv=8,
    n_layers=80,
    n_rep_kv=8,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
)
