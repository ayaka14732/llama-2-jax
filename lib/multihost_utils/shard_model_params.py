import jax

from ..LLM import Llama, LlamaModel
from ..LLM.attention import Attention
from ..LLM.decoder import Decoder
from .shard_array import shard_array

sharding_mp = Llama(
    model=LlamaModel(
        embedding=...,
        decoder=Decoder(
            input_norm=...,
            attention=Attention(q_proj=3, k_proj=2, v_proj=2, out_proj=2),
            post_attn_norm=...,
            gate_proj=2,
            up_proj=2,
            down_proj=1,
        ),
        norm=...,
    ),
    lm_head=...,
)

def shard_model_params(params: Llama) -> Llama:
    return jax.tree_map(shard_array, params, sharding_mp)
