# JAX Implementation of LLaMA

This project is still in progress. Please check it out in the next week.

This project is a JAX implementation of [LLaMA](https://arxiv.org/abs/1910.13461). This project aims to train LLaMA on Google Cloud TPU.

This project is inspired by [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax).

## Environment Setup

This project requires at least Python 3.11, JAX 0.4.11, PyTorch 2.1.0 and Transformers 4.30.0.

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
