# JAX Implementation of Llama 2

This project is the JAX implementation of [Llama 2](https://arxiv.org/abs/1910.13461).

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

This project is inspired by [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax).

## Motivation

The objectives of this project are threefold:

- Implement the Llama 2 model using JAX to enable efficient training and inference on Google Cloud TPU;
- Develop a high-quality codebase that serves as an exemplary implementation of the Transformer model using JAX;
- Facilitate the identification of common errors and inconsistencies across various transformer models through the implementation of a high-quality codebase, thereby providing valuable insights for the NLP community.

## Roadmap

- [x] [Parameter conversion](lib/param_utils/convert_params.py)
- [x] Model architecture
    - [x] [Dropout](lib/model/dropout.py)
    - [x] [RMS Norm](lib/model/rms_norm.py)
    - [x] [Embedding](lib/model/embedding.py)
    - [x] [Rotary embedding](lib/model/rotary_embedding.py)
    - [x] [Attention](lib/model/attention.py)
    - [x] [Decoder block](lib/model/decoder_block.py)
    - [x] [Decoder](lib/model/decoder.py)
    - [x] [LLaMA Model](lib/model/llama_model.py)
- [x] [Cross entropy loss](lib/loss/cross_entropy_loss.py)
- [x] Logits processing
    - [x] [Bias](lib/logits_processing/bias.py)
    - [x] [Penalize presence](lib/logits_processing/penalize_presence.py)
    - [x] [Penalize frequency](lib/logits_processing/penalize_frequency.py)
- [ ] Generation
    - [ ] Beam search
    - [ ] Beam sampling
    - [x] [Top-_k_ sampling](lib/generation/top_k.py)
    - [x] [Top-_p_ sampling](lib/generation/top_p.py)
- [x] [Data loading](lib/dataloader/LlamaDataLoader.py)
- [ ] Distributed and parallel inference
- [ ] Distributed and parallel training

## Environment Setup

This project requires at least Python 3.11, JAX 0.4.13, PyTorch 2.1.0 and Transformers 4.31.0.dev0.

PyTorch and Transformers are needed for testing purposes. Additionally, the data loader depends on PyTorch `DataLoader`, while the profiling functionality requires TensorFlow.

### Create `venv`

```sh
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
```

### Install the proper version of JAX

CUDA 11.8:

```sh
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

TPU:

```sh
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Install other dependencies

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

### Download LLaMA weights

You can request to access the Llama weights from [the official website](https://ai.meta.com/llama/).

### Convert parameters

```sh
(cd .. && git clone https://github.com/huggingface/transformers.git)
python ../transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama-weights-original --model_size 7B --output_dir ../llama-weights/7B
python scripts/convert_params_runner.py
```

### Test generation

```sh
python generate.py
```

## Model Configurations

| Name | Parameters | `n_layers` | `n_heads_q` | `n_rep_kv` | `d_model`| `d_ff` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 7B |  | 32 | 32 | 1 | 4096 | 11008 |
| 13B | | 40 | 40 | 1 | 5120 | |
| 70B | | 80 | 64 | | 8192 | |

## Model Architecture

The Hugging Face format is like this:

```
LlamaForCausalLM(
  (model): LlamaModel(
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
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

The format used in this project is like this:

```
model
  embedding: (32000, 4096)
  decoder: 32 x decoder_block
    input_norm: (4096)
    attention
      q_proj: (4096, 1, 32, 128)
      k_proj: (4096, 32, 128)
      v_proj: (4096, 32, 128)
      out_proj: (1, 32, 128, 4096)
    post_attn_norm: (4096)
    gate_proj: (4096, 11008)
    up_proj: (4096, 11008)
    down_proj: (11008, 4096)
  norm: (4096)
lm_head: (4096, 32000)
```
