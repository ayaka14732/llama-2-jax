from jax import Array
import torch
from transformers import LlamaModel

from lib.array_utils import pt2jax
from lib.model import Attention, Config, DecoderBlock, Llama

def convert_proj(x) -> Array:
    return pt2jax(x.weight.T)

def convert_qk_proj(x, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.d_model, config.n_heads, config.d_k))

def convert_v_proj(x, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.d_model, config.n_heads, config.d_v))

def convert_out_proj(x, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.n_heads, config.d_v, config.d_model))

def convert_attention(x, *, config: Config) -> Attention:
    q_proj = convert_qk_proj(x.q_proj, config=config)
    k_proj = convert_qk_proj(x.k_proj, config=config)
    v_proj = convert_v_proj(x.v_proj, config=config)
    out_proj = convert_out_proj(x.o_proj, config=config)
    return Attention(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, out_proj=out_proj)

def convert_decoder_block(x, *, config: Config) -> DecoderBlock:
    input_norm = pt2jax(x.input_layernorm.weight)
    attention = convert_attention(x.self_attn, config=config)
    post_attn_norm = pt2jax(x.post_attention_layernorm.weight)
    gate_proj = convert_proj(x.mlp.gate_proj)
    up_proj = convert_proj(x.mlp.up_proj)
    down_proj = convert_proj(x.mlp.down_proj)
    return DecoderBlock(input_norm=input_norm, attention=attention, post_attn_norm=post_attn_norm, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)

def convert_params(model_pt: LlamaModel, *, config: Config) -> Llama:
    with torch.no_grad():
        embedding = pt2jax(model_pt.embed_tokens.weight)
        decoder = [convert_decoder_block(model_pt.layers[i], config=config) for i in range(config.n_layers)]
        norm = pt2jax(model_pt.norm.weight)
        return Llama(embedding=embedding, decoder=decoder, norm=norm)
