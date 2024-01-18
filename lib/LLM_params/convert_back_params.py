from jax import Array
import torch
import torch.nn as tnn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel as LlamaModelPt
from transformers import MistralConfig, MistralForCausalLM, MistralModel as MistralModelPt
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralDecoderLayer, MistralMLP, MistralRMSNorm

from ..array_utils import jax2pt
from ..LLM import Llama, LlamaModel
from ..LLM.attention import Attention
from ..LLM.decoder_block import DecoderBlock
from ..tree_utils import unstack_leaves

def convert_back_embedding(x: Array) -> tnn.Embedding:
    with torch.no_grad():
        embedding = tnn.Embedding(*x.shape)  # type: ignore
        embedding.weight = tnn.Parameter(jax2pt(x))
        return embedding

def convert_back_norm(x: Array, *, config: LlamaConfig) -> LlamaRMSNorm:
    d_model = config.hidden_size
    rms_norm_eps = config.rms_norm_eps
    with torch.no_grad():
        llama_rms_norm = LlamaRMSNorm(d_model, eps=rms_norm_eps)
        llama_rms_norm.weight = tnn.Parameter(jax2pt(x))
        return llama_rms_norm
    
def convert_back_mistral_norm(x: Array, *, config: MistralConfig) -> MistralRMSNorm:
    d_model = config.hidden_size
    rms_norm_eps = config.rms_norm_eps
    with torch.no_grad():
        mistral_rms_norm = MistralRMSNorm(d_model, eps=rms_norm_eps)
        mistral_rms_norm.weight = tnn.Parameter(jax2pt(x))
        return mistral_rms_norm

def convert_back_proj(x: Array) -> tnn.Linear:
    with torch.no_grad():
        linear = tnn.Linear(*x.shape, bias=False)  # type: ignore
        linear.weight = tnn.Parameter(jax2pt(x).T)
        return linear
    
def convert_back_mistral_proj(x: Array) -> tnn.Linear:
    with torch.no_grad():
        linear = tnn.Linear(*x.shape, bias=False)  # type: ignore
        linear.weight = tnn.Parameter(jax2pt(x).T)
        return linear

def convert_back_q_proj(x: Array, *, config: LlamaConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_rep_kv = config.num_attention_heads // config.num_key_value_heads
    n_heads_kv = config.num_key_value_heads
    d_k = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_rep_kv * n_heads_kv * d_k
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_mistral_q_proj(x: Array, *, config: MistralConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_rep_kv = config.num_attention_heads // config.num_key_value_heads
    n_heads_kv = config.num_key_value_heads
    d_k = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_rep_kv * n_heads_kv * d_k
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear


def convert_back_k_proj(x: Array, *, config: LlamaConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_heads_kv = config.num_key_value_heads
    d_k = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_heads_kv * d_k
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_mistral_k_proj(x: Array, *, config: MistralConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_heads_kv = config.num_key_value_heads
    d_k = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_heads_kv * d_k
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_v_proj(x: Array, *, config: LlamaConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_heads_kv = config.num_key_value_heads
    d_v = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_heads_kv * d_v
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear
    
def convert_back_mistral_v_proj(x: Array, *, config: MistralConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_heads_kv = config.num_key_value_heads
    d_v = config.hidden_size // config.num_attention_heads
    in_features = d_model
    out_features = n_heads_kv * d_v
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_out_proj(x: Array, *, config: LlamaConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_rep_kv = config.num_attention_heads // config.num_key_value_heads
    n_heads_kv = config.num_key_value_heads
    d_v = config.hidden_size // config.num_attention_heads
    in_features = n_rep_kv * n_heads_kv * d_v
    out_features = d_model
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)  # type: ignore
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_mistral_out_proj(x: Array, *, config: MistralConfig) -> tnn.Linear:
    d_model = config.hidden_size
    n_rep_kv = config.num_attention_heads // config.num_key_value_heads
    n_heads_kv = config.num_key_value_heads
    d_v = config.hidden_size // config.num_attention_heads
    in_features = n_rep_kv * n_heads_kv * d_v
    out_features = d_model
    with torch.no_grad():
        linear = tnn.Linear(in_features, out_features, bias=False)  # type: ignore
        linear.weight = tnn.Parameter(jax2pt(x).reshape(in_features, out_features).T)
        return linear

def convert_back_attention(x: Attention, *, config: LlamaConfig) -> LlamaAttention:
    with torch.no_grad():
        llama_attention = LlamaAttention(config=config)
        llama_attention.q_proj = convert_back_q_proj(x.q_proj, config=config)
        llama_attention.k_proj = convert_back_k_proj(x.k_proj, config=config)
        llama_attention.v_proj = convert_back_v_proj(x.v_proj, config=config)
        llama_attention.o_proj = convert_back_out_proj(x.out_proj, config=config)
        return llama_attention
    
def convert_back_mistral_attention(x: Attention, *, config: MistralConfig) -> MistralAttention:
    with torch.no_grad():
        mistral_attention = MistralAttention(config=config)
        mistral_attention.q_proj = convert_back_mistral_q_proj(x.q_proj, config=config)
        mistral_attention.k_proj = convert_back_mistral_k_proj(x.k_proj, config=config)
        mistral_attention.v_proj = convert_back_mistral_v_proj(x.v_proj, config=config)
        mistral_attention.o_proj = convert_back_mistral_out_proj(x.out_proj, config=config)
        return mistral_attention

def convert_back_mlp(gate_proj: Array, up_proj: Array, down_proj: Array, *, config: LlamaConfig) -> LlamaMLP:
    with torch.no_grad():
        llama_mlp = LlamaMLP(config=config)
        llama_mlp.gate_proj = convert_back_proj(gate_proj)
        llama_mlp.up_proj = convert_back_proj(up_proj)
        llama_mlp.down_proj = convert_back_proj(down_proj)
        return llama_mlp
    
def convert_back_mistral_mlp(gate_proj: Array, up_proj: Array, down_proj: Array, *, config: MistralConfig) -> MistralMLP:
    with torch.no_grad():
        mistral_mlp = MistralMLP(config=config)
        mistral_mlp.gate_proj = convert_back_mistral_proj(gate_proj)
        mistral_mlp.up_proj = convert_back_mistral_proj(up_proj)
        mistral_mlp.down_proj = convert_back_mistral_proj(down_proj)
        return mistral_mlp
    

def convert_back_decoder_block(x: DecoderBlock, *, config: LlamaConfig) -> LlamaDecoderLayer:
    with torch.no_grad():
        llama_decoder_layer = LlamaDecoderLayer(config=config)
        llama_decoder_layer.self_attn = convert_back_attention(x.attention, config=config)
        llama_decoder_layer.mlp = convert_back_mlp(x.gate_proj, x.up_proj, x.down_proj, config=config)
        llama_decoder_layer.input_layernorm = convert_back_norm(x.input_norm, config=config)
        llama_decoder_layer.post_attention_layernorm = convert_back_norm(x.post_attn_norm, config=config)
        return llama_decoder_layer
    
def convert_back_mistral_decoder_block(x: DecoderBlock, *, config: MistralConfig) -> MistralDecoderLayer:
    with torch.no_grad():
        mistral_decoder_layer = MistralDecoderLayer(config=config)
        mistral_decoder_layer.self_attn = convert_back_mistral_attention(x.attention, config=config)
        mistral_decoder_layer.mlp = convert_back_mistral_mlp(x.gate_proj, x.up_proj, x.down_proj, config=config)
        mistral_decoder_layer.input_layernorm = convert_back_mistral_norm(x.input_norm, config=config)
        mistral_decoder_layer.post_attention_layernorm = convert_back_mistral_norm(x.post_attn_norm, config=config)
        return mistral_decoder_layer
    

def convert_back_llama_model(x: LlamaModel, *, config: LlamaConfig) -> LlamaModelPt:
    with torch.no_grad():
        llama_model = LlamaModelPt(config=config)
        llama_model.embed_tokens = convert_back_embedding(x.embedding)
        llama_model.layers = tnn.ModuleList([convert_back_decoder_block(decoder_block, config=config) for decoder_block in unstack_leaves(x.decoder)])
        llama_model.norm = convert_back_norm(x.norm, config=config)
        return llama_model
    
def convert_back_mistral_model(x: LlamaModel, *, config: MistralConfig) -> MistralModelPt:
    with torch.no_grad():
        mistral_model = MistralModelPt(config=config)
        mistral_model.embed_tokens = convert_back_embedding(x.embedding)
        mistral_model.layers = tnn.ModuleList([convert_back_mistral_decoder_block(decoder_block, config=config) for decoder_block in unstack_leaves(x.decoder)])
        mistral_model.norm = convert_back_mistral_norm(x.norm, config=config)
        return mistral_model
    
    
def convert_back_llama(x: Llama, *, config: LlamaConfig) -> LlamaForCausalLM:
    with torch.no_grad():
        llama = LlamaForCausalLM(config=config)
        llama.model = convert_back_llama_model(x.model, config=config)
        llama.lm_head = convert_back_proj(x.lm_head)
        return llama
    
def convert_back_mistral(x: Llama, *, config: MistralConfig) -> MistralForCausalLM:
    with torch.no_grad():
        mistral = MistralForCausalLM(config=config)
        mistral.model = convert_back_mistral_model(x.model, config=config)
        mistral.lm_head = convert_back_mistral_proj(x.lm_head)
        return mistral

# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# from lib.proc_init_utils import initialise_cpu; initialise_cpu()
# model_pt = LlamaForCausalLM.from_pretrained('/dev/shm/llama-weights/llama2-7B')
# config = LlamaConfig.from_pretrained('/dev/shm/llama-weights/llama2-7B')
# from lib.param_utils.convert_params import convert_proj
# assert torch.equal(convert_back_proj(convert_proj(model_pt.lm_head)).weight, model_pt.lm_head.weight)
# assert torch.equal(convert_back_proj(convert_proj(model_pt.model.layers[0].self_attn.q_proj)).weight, model_pt.model.layers[0].self_attn.q_proj.weight)
# assert torch.equal(convert_back_proj(convert_proj(model_pt.model.layers[0].self_attn.k_proj)).weight, model_pt.model.layers[0].self_attn.k_proj.weight)
# assert torch.equal(convert_back_proj(convert_proj(model_pt.model.layers[0].self_attn.v_proj)).weight, model_pt.model.layers[0].self_attn.v_proj.weight)
# assert torch.equal(convert_back_proj(convert_proj(model_pt.model.layers[0].self_attn.o_proj)).weight, model_pt.model.layers[0].self_attn.o_proj.weight)
# model_pt.model.norm.weight
# model_pt.model.embed_tokens.weight
