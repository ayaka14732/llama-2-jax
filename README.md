# JAX Implementation of LLaMA

This project is still in progress. Please check it out in the next week.

This project is a JAX implementation of [LLaMA](https://arxiv.org/abs/1910.13461). This project aims to train LLaMA on Google Cloud TPU.

This project is inspired by [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax).

## Environment Setup

This project requires at least Python 3.11, JAX 0.4.12, PyTorch 2.1.0 and Transformers 4.31.0.

PyTorch and Transformers are needed for testing purposes.

```sh
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

## Model Configurations

| Name | Parameters | `n_layers` | `n_heads` | `d_model`| `d_ff` |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 7B | 6,607,343,616 | 32 | 32 | 4096 | 11008 |
| 13B | | 40 | 40 | 5120 | |
| 30B* | | 60 | 52 | 6656 | |
| 65B | | 80 | 64 | 8192 | |

\* The model name is 30B, but the actual model size is 33B.

## Model Architecture

The Hugging Face format is like this:

```
LlamaModel(
  (embed_tokens): Embedding(32000, 4096, padding_idx=0)
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (act_fn): SiLUActivation()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
```

The format used in this project is like this:

```
embedding: (32000, 4096)
decoder: 32 x decoder_block
  input_norm: (4096)
  attention
    q_proj: (4096, 32, 128)
    k_proj: (4096, 32, 128)
    v_proj: (4096, 32, 128)
    out_proj: (32, 128, 4096)
  post_attn_norm: (4096)
  gate_proj: (4096, 11008)
  up_proj: (4096, 11008)
  down_proj: (11008, 4096)
norm: (4096)
```
