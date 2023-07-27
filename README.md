# JAX Implementation of Llama 2

This project is the JAX implementation of [Llama 2](https://arxiv.org/abs/1910.13461).

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

This project is a direct fork of [ayaka14732/llama-jax](https://github.com/ayaka14732/llama-jax), which is inspired by [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax).

## Motivation

The objectives of this project are threefold:

- Implement the Llama 2 model using JAX to enable efficient training and inference on Google Cloud TPU;
- Develop a high-quality codebase that serves as an exemplary implementation of the Transformer model using JAX;
- Facilitate the identification of common errors and inconsistencies across various transformer models through the implementation of a high-quality codebase, thereby providing valuable insights for the NLP community.

## Features

- [x] Parameter conversion
    - [x] [Convert from Hugging Face](lib/param_utils/convert_params.py)
    - [x] [Convert to Hugging Face](lib/param_utils/convert_back_params.py)
- [x] Model architecture
    - [x] [Dropout](lib/model/dropout.py)
    - [x] [RMS Norm](lib/model/rms_norm.py)
    - [x] [Embedding](lib/model/embedding.py)
    - [x] [Rotary embedding](lib/model/rotary_embedding.py)
    - [x] [Attention](lib/model/attention.py)
    - [x] [Decoder block](lib/model/decoder_block.py)
    - [x] [Decoder](lib/model/decoder.py)
    - [x] [Llama Model](lib/model/llama_model.py)
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
    - [ ] Optimisation
- [x] [Data loading](lib/dataloader/LlamaDataLoader.py)
- [x] Inference
- [x] Training
- [x] Parallelisation
    - [x] [Model parallelism](lib/multihost_utils/shard_model_params_to_multihost.py)

## Environment Setup

This project requires at least Python 3.11, JAX 0.4.13, PyTorch 2.1.0 and Transformers 4.32.0.dev0.

PyTorch and Transformers are needed for testing purposes. Additionally, the data loader depends on PyTorch `DataLoader`, while the profiling functionality requires TensorFlow.

### Install Python 3.11

For Ubuntu users, you can follow [How to install Python 3.11 on Ubuntu 22.04](https://unixcop.com/install-python-3-11-ubuntu/) to Install Python 3.11. The tutorial applied to Ubuntu 20.04 as well.

### Create `venv`

```sh
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
```

### Install the proper version of JAX

You need to follow the installation instructions on JAX's [offical GitHub page](https://github.com/google/jax#installation).

TPU:

```sh
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

CUDA 12:

```sh
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

CUDA 11.8:

```sh
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install the proper version of PyTorch

Typically, you only need to install the CPU version of PyTorch since we perform most of the computation using JAX. However, it's worth noting that the current codebase's generation process is not fully optimised yet. To expedite the inference, one effective approach would involve converting the model back to Hugging Face format and running the inference in PyTorch.

To install PyTorch, you can follow the [official installation guide](https://pytorch.org/get-started/locally/).

CPU:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

CUDA 12:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

CUDA 11.8:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
```

### Install other dependencies

```sh
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

### Download LLaMA weights

LLaMA 1:

If you couldn't obtain the LLaMA weights, you can download them with [shawwn/llama-dl](https://github.com/shawwn/llama-dl).

```sh
mkdir ../llama-weights-original && cd ../llama-weights-original
curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | bash
python ../llama-2-jax/venv/lib/python3.11/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama-weights-original --model_size 7B --output_dir ../llama-weights/7B
```

Llama 2:

You can request to access the Llama weights from [the official website](https://ai.meta.com/llama/).

```sh
cd .. && git clone --depth=1 https://github.com/facebookresearch/llama llama-meta
cd llama-meta
chmod +x download.sh
./download.sh

ln -sf llama-2-7b 7Bf
python ../llama-2-jax/venv/lib/python3.11/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama-meta --model_size 7Bf --output_dir ../llama-weights/llama2-7B

ln -sf llama-2-70b 70Bf
python ../llama-2-jax/venv/lib/python3.11/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama-meta --model_size 70Bf --output_dir ../llama-weights/llama2-70B
```

### Convert parameters

```sh
python scripts/convert_params_runner.py llama1-7B
python scripts/convert_params_runner.py llama2-7B
python scripts/convert_params_runner.py llama2-70B
```

### Special configuration for TPU Pods

If you are running on TPU pods or other multi-host environments, you need to put the IP address of other machines in `external-ips.txt` (one IP address per line). Besides, you should make sure that one of the hosts can SSH into other hosts.

### Generation

```sh
python generate.py
```

On TPU pods, the command is:

```sh
./startpod python generate.py
```

## Training

I present a simple example of the training pipeline by fine-tuning the model on the GSM dataset.

### Download GSM dataset

```sh
cd .. && git clone --depth=1 https://github.com/openai/grade-school-math.git
```

### Run the training script

```sh
python train.py
```

On TPU pods, the command is:

```sh
./startpod python train.py
```

## Model Configurations

| Name | Parameters | `vocab_size` | `n_layers` | `n_heads_kv` | `n_rep_kv` | `d_model` | `d_ff` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| LLaMA 1 7B | 6738415616 | 32000 | 32 | 32 | 1 | 4096 | 11008 |
| LLaMA 1 13B | | 32000 | 40 | 40 | 1 | 5120 | |
| LLaMA 1 33B | | 32000 | 60 | 52 | 1 | 6656 | |
| LLaMA 1 65B | | 32000 | 80 | 64 | 1 | 8192 | |
| Llama 2 7B | 6738415616 | 32000 | 32 | 32 | 1 | 4096 | 11008 |
| Llama 2 13B | | 32000 | | | | | |
| Llama 2 70B | | 32000 | 80 | 8 | 8 | 8192 | 28672 |

```
  n_parameters
= 2 * vocab_size * d_model
+ (2 * n_layers + 1) * d_model
+ 2 * n_layers * d_model * n_rep_kv * n_heads_kv * d_k
+ 2 * n_layers * d_model * n_heads_kv * d_k
+ 3 * n_layers * d_model * d_ff
```

## Model Architecture

### LLaMA 1 (7B)

Hugging Face format:

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

The format used in this project:

```
model
  embedding: (32000, 4096)
  decoder: decoder_block
    input_norm: (32, 4096)
    attention
      q_proj: (32, 4096, 1, 32, 128)
      k_proj: (32, 4096, 32, 128)
      v_proj: (32, 4096, 32, 128)
      out_proj: (32, 1, 32, 128, 4096)
    post_attn_norm: (32, 4096)
    gate_proj: (32, 4096, 11008)
    up_proj: (32, 4096, 11008)
    down_proj: (32, 11008, 4096)
  norm: (4096)
lm_head: (4096, 32000)
```

### Llama 2 (70B)

Hugging Face format:

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 8192, padding_idx=0)
    (layers): ModuleList(
      (0-79): 80 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
          (k_proj): Linear(in_features=8192, out_features=1024, bias=False)
          (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
          (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=8192, out_features=28672, bias=False)
          (up_proj): Linear(in_features=8192, out_features=28672, bias=False)
          (down_proj): Linear(in_features=28672, out_features=8192, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=8192, out_features=32000, bias=False)
)
```

The format used in this project:

```
model
  embedding: (32000, 8192)
  decoder: decoder_block
    input_norm: (80, 8192)
    attention
      q_proj: (80, 8192, 8, 8, 128)
      k_proj: (80, 8192, 8, 128)
      v_proj: (80, 8192, 8, 128)
      out_proj: (80, 8, 8, 128, 8192)
    post_attn_norm: (80, 8192)
    gate_proj: (80, 8192, 28672)
    up_proj: (80, 8192, 28672)
    down_proj: (80, 28672, 8192)
  norm: (8192)
lm_head: (8192, 32000)
```

## Findings

- LLaMA utilises rotary positional embeddings.
- There is no bias in the Q, K, V matrices and the linear projections in the FFNs, which is the same as the original transformer, but different from BERT and BART.
- In Llama models, each FFN has 3 linear projections, while in BART there are only 2.
- There is no dropout in the original LLaMA implementation.
- Llama 2 70B utilises Grouped-Query Attention (GQA).
