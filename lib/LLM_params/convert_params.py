from jax import Array
import torch
import torch.nn as tnn
from transformers import LlamaForCausalLM, LlamaModel as LlamaModelPt
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
import gc
from ..array_utils import pt2jax
from ..LLM import Llama, LlamaModel, ModelConfig
from ..LLM.attention import Attention
from ..LLM.decoder_block import DecoderBlock
from ..tree_utils import stack_leaves

def convert_proj(x: tnn.Linear) -> Array:
    return pt2jax(x.weight.T)

def convert_q_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    return pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))

def convert_k_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    return pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_heads_kv, model_config.d_k))

def convert_v_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    return pt2jax(x.weight.T.reshape(model_config.d_model, model_config.n_heads_kv, model_config.d_v))

def convert_out_proj(x: tnn.Linear, *, model_config: ModelConfig) -> Array:
    return pt2jax(x.weight.T.reshape(model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))

def convert_attention(x: LlamaAttention, *, model_config: ModelConfig) -> Attention:
    q_proj = convert_q_proj(x.q_proj, model_config=model_config)
    k_proj = convert_k_proj(x.k_proj, model_config=model_config)
    v_proj = convert_v_proj(x.v_proj, model_config=model_config)
    out_proj = convert_out_proj(x.o_proj, model_config=model_config)
    gc.collect()
    return Attention(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, out_proj=out_proj)

def convert_decoder_block(x: LlamaDecoderLayer, *, model_config: ModelConfig) -> DecoderBlock:
    input_norm = pt2jax(x.input_layernorm.weight)
    attention = convert_attention(x.self_attn, model_config=model_config)
    post_attn_norm = pt2jax(x.post_attention_layernorm.weight)
    gate_proj = convert_proj(x.mlp.gate_proj)
    up_proj = convert_proj(x.mlp.up_proj)
    down_proj = convert_proj(x.mlp.down_proj)
    gc.collect()
    return DecoderBlock(input_norm=input_norm, attention=attention, post_attn_norm=post_attn_norm, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)

def convert_llama_model(model: LlamaModelPt, *, model_config: ModelConfig) -> LlamaModel:
    embedding = pt2jax(model.embed_tokens.weight)
    decoder = stack_leaves([convert_decoder_block(model.layers[i], model_config=model_config) for i in range(model_config.n_layers)])
    norm = pt2jax(model.norm.weight)
    gc.collect()
    return LlamaModel(embedding=embedding, decoder=decoder, norm=norm)

def convert_llama(model_pt: LlamaForCausalLM, *, model_config: ModelConfig) -> Llama:
    with torch.no_grad():
        model = convert_llama_model(model_pt.model, model_config=model_config)
        lm_head = convert_proj(model_pt.lm_head)
        return Llama(model=model, lm_head=lm_head)
