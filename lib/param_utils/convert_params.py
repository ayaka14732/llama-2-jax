from jax import Array
import torch
import torch.nn as tnn
from transformers import LlamaForCausalLM, LlamaModel as LlamaModelPt
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer

from lib.array_utils import pt2jax
from lib.model import Attention, Config, DecoderBlock, Llama, LlamaModel

def convert_proj(x: tnn.Linear) -> Array:
    return pt2jax(x.weight.T)

def convert_q_proj(x: tnn.Linear, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.d_model, config.n_rep_kv, config.n_heads_kv, config.d_k))

def convert_k_proj(x: tnn.Linear, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.d_model, config.n_heads_kv, config.d_k))

def convert_v_proj(x: tnn.Linear, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.d_model, config.n_heads_kv, config.d_v))

def convert_out_proj(x: tnn.Linear, *, config: Config) -> Array:
    return pt2jax(x.weight.T.reshape(config.n_rep_kv, config.n_heads_kv, config.d_v, config.d_model))

def convert_attention(x: LlamaAttention, *, config: Config) -> Attention:
    q_proj = convert_q_proj(x.q_proj, config=config)
    k_proj = convert_k_proj(x.k_proj, config=config)
    v_proj = convert_v_proj(x.v_proj, config=config)
    out_proj = convert_out_proj(x.o_proj, config=config)
    return Attention(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, out_proj=out_proj)

def convert_decoder_block(x: LlamaDecoderLayer, *, config: Config) -> DecoderBlock:
    input_norm = pt2jax(x.input_layernorm.weight)
    attention = convert_attention(x.self_attn, config=config)
    post_attn_norm = pt2jax(x.post_attention_layernorm.weight)
    gate_proj = convert_proj(x.mlp.gate_proj)
    up_proj = convert_proj(x.mlp.up_proj)
    down_proj = convert_proj(x.mlp.down_proj)
    return DecoderBlock(input_norm=input_norm, attention=attention, post_attn_norm=post_attn_norm, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)

def convert_llama_model(model: LlamaModelPt, *, config: Config) -> LlamaModel:
    embedding = pt2jax(model.embed_tokens.weight)
    decoder = [convert_decoder_block(model.layers[i], config=config) for i in range(config.n_layers)]
    norm = pt2jax(model.norm.weight)
    return LlamaModel(embedding=embedding, decoder=decoder, norm=norm)

def convert_llama(model_pt: LlamaForCausalLM, *, config: Config) -> Llama:
    with torch.no_grad():
        model = convert_llama_model(model_pt.model, config=config)
        lm_head = convert_proj(model_pt.lm_head)
        return Llama(model=model, lm_head=lm_head)
