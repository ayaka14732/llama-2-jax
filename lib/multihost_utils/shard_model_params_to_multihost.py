from ..model import Attention, Decoder, Llama, LlamaModel
from ..tree_utils import tree_apply
from .shard_array_to_multihost import shard_array_to_multihost

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

def shard_model_params_to_multihost(params: Llama) -> Llama:
    return tree_apply(shard_array_to_multihost, params, sharding_mp)
