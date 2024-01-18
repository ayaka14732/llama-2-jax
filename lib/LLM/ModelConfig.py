from typing import NamedTuple

class ModelConfig(NamedTuple):
    d_ff: int
    d_k: int
    d_model: int
    d_v: int
    n_heads_kv: int
    n_layers: int
    n_rep_kv: int
    rms_norm_eps: float
    token_id_bos: int
    token_id_eos: int
    token_id_pad: int
    vocab_size: int

    # TODO: move out of model config
    dropout_rate: float | None
    return_kv_cache: bool

model_config_dummy = ModelConfig(
    d_ff=32,
    d_k=32,
    d_model=15,
    d_v=16,
    n_heads_kv=32,
    n_layers=3,
    n_rep_kv=2,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_llama1_7B = ModelConfig(
    d_ff=11008,
    d_k=128,
    d_model=4096,
    d_v=128,
    n_heads_kv=32,
    n_layers=32,
    n_rep_kv=1,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_llama1_13B = ModelConfig(
    d_ff=13824,
    d_k=128,
    d_model=5120,
    d_v=128,
    n_heads_kv=40,
    n_layers=40,
    n_rep_kv=1,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_llama1_30B = ModelConfig(
    d_ff=17920,
    d_k=128,
    d_model=6656,
    d_v=128,
    n_heads_kv=52,
    n_layers=60,
    n_rep_kv=1,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_llama2_7B = model_config_llama1_7B

model_config_llama2_13B = ModelConfig(
    d_ff=13824,
    d_k=128,
    d_model=5120,
    d_v=128,
    n_heads_kv=40,
    n_layers=40,
    n_rep_kv=1,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_llama2_70B = ModelConfig(
    d_ff=28672,
    d_k=128,
    d_model=8192,
    d_v=128,
    n_heads_kv=8,
    n_layers=80,
    n_rep_kv=8,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_orca2_7B = ModelConfig(
    d_ff=11008,
    d_k=128,
    d_model=4096,
    d_v=128,
    n_heads_kv=32,
    n_layers=32,
    n_rep_kv=1,
    rms_norm_eps=1e-5,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32003,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_orca2_13B = ModelConfig(
    d_ff=13824,
    d_k=128,
    d_model=5120,
    d_v=128,
    n_heads_kv=40,
    n_layers=40,
    n_rep_kv=1,
    rms_norm_eps=1e-5,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32003,
    dropout_rate=0.1,
    return_kv_cache=False,
)

model_config_solar_10_7B = ModelConfig(
    d_ff=14336,
    d_k=128,
    d_model=4096,
    d_v=128,
    n_heads_kv=8,
    n_layers=48,
    n_rep_kv=4,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)
mistral_config_7B = ModelConfig(
    d_ff=11008,
    d_k=128,
    d_model=4096,
    d_v=128,
    n_heads_kv=8,
    n_layers=32,
    n_rep_kv=4,
    rms_norm_eps=1e-6,
    token_id_bos=1,
    token_id_eos=2,
    token_id_pad=0,
    vocab_size=32000,
    dropout_rate=0.1,
    return_kv_cache=False,
)

zephyr_config_3B = ModelConfig(
    d_ff=6912,
    d_k=80,
    d_model=2560,
    d_v=80,
    n_heads_kv=32,
    n_layers=32,
    n_rep_kv=1,
    rms_norm_eps=1e-5,
    token_id_bos=0,
    token_id_eos=0,
    token_id_pad=0,
    vocab_size=50304,
    dropout_rate=0.1,
    return_kv_cache=True,
)
