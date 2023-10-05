# Llama 2 JAX

This project is the JAX implementation of [Llama 2](https://arxiv.org/abs/1910.13461). The objectives of this project are threefold:

1. Implement the Llama 2 model using JAX to enable efficient training and inference on Google Cloud TPU;
1. Develop a high-quality codebase that serves as an exemplary implementation of the Transformer model using JAX;
1. Facilitate the identification of common errors and inconsistencies across various transformer models through the implementation of a high-quality codebase, thereby providing valuable insights for the NLP community.

Related projects:

- [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer): PyTorch implementation of the original Transformer
- [ayaka14732/TrAVis](https://github.com/ayaka14732/TrAVis): An in-browser Transformer attention visualiser that includes a NumPy implementation of BERT
- [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax): JAX implementation of BART-base
- [ztjhz/t5-jax](https://github.com/ztjhz/t5-jax): JAX implementation of T5
- [young-geng/EasyLM](https://github.com/young-geng/EasyLM): LLM framework that includes Flax implementations of LLaMA, GPT-J and RoBERTa

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Features

- [x] [Parameter conversion](lib/llama_params/)
    - [x] [Hugging Face to JAX](lib/llama_params/convert_params.py)
    - [x] [JAX to Hugging Face](lib/llama_params/convert_back_params.py)
- [x] [Model architecture](lib/llama/)
    - [x] [Dropout](lib/llama/dropout.py)
    - [x] [RMS Norm](lib/llama/rms_norm.py)
    - [x] [Embedding](lib/llama/embedding.py)
    - [x] [Rotary embedding](lib/llama/rotary_embedding.py)
    - [x] [Attention](lib/llama/attention.py)
    - [x] [Decoder block](lib/llama/decoder_block.py)
    - [x] [Decoder](lib/llama/decoder.py)
    - [x] [Llama Model](lib/llama/llama_model.py)
- [x] [Cross entropy loss](lib/loss/cross_entropy_loss.py)
- [x] [Logits processing](lib/logits_processing/)
    - [x] [Bias](lib/logits_processing/bias.py)
    - [x] [Penalize presence](lib/logits_processing/penalize_presence.py)
    - [x] [Penalize frequency](lib/logits_processing/penalize_frequency.py)
- [ ] Generation
    - [x] KV cache
    - [ ] Beam search
    - [ ] Beam sampling
    - [x] [Top-_k_ sampling](lib/generation/top_k.py)
    - [x] [Top-_p_ sampling](lib/generation/top_p.py)
- [x] [Data loading](lib/dataloader/LlamaDataLoader.py)
- [x] Inference
- [x] Training
- [x] Parallelisation
    - [x] [Model parallelism](lib/multihost_utils/shard_model_params_to_multihost.py)
- [ ] Documentation

The documentation of the library of this project is published on [GitHub Pages](https://ayaka14732.github.io/llama-2-jax/).

## Environment Setup

This project requires at least Python 3.11, JAX 0.4.17, PyTorch 2.1.0, Optax 0.1.8.dev0 and Transformers 4.32.0.dev0.

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

### Special configuration for TPU Pods

If you are running on TPU pods, you need to put the IP address of all other hosts in `~/podips.txt` (one IP address per line). Besides, you should make sure that the local host can SSH into itself and all other hosts listed in the file.

### Install the proper version of JAX

You need to follow the installation instructions on JAX's [official GitHub page](https://github.com/google/jax#installation).

### Install the proper version of PyTorch

Typically, you only need to install the CPU version of PyTorch since we perform most of the computation using JAX. However, it's worth noting that the current codebase's generation process is not fully optimised yet. To expedite the inference, one effective approach would involve converting the model back to Hugging Face format and running the inference in PyTorch.

To install PyTorch, you can follow the [official installation guide](https://pytorch.org/get-started/locally/).

On TPU VMs, this is usually:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

On TPU Pods:

```sh
./podrun -i -- ~/venv/bin/pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Install other dependencies

```sh
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/deepmind/optax.git  # https://github.com/google-deepmind/optax/issues/472
pip install -r requirements.txt
```

On TPU Pods:

```sh
./podrun -i -- ~/venv/bin/pip install git+https://github.com/huggingface/transformers.git
./podrun -i -- ~/venv/bin/pip install git+https://github.com/deepmind/optax.git
./podrun -iw -- ~/venv/bin/pip install -r requirements.txt
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

You can request to access the Llama weights from [the official website](https://ai.meta.com/llama/). After your request is approved, you will automatically get access to the Hugging Face Llama 2 models. You can verify that the models are accessible by trying to access the [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) version.

### Login into Hugging Face CLI

If you need to work with Llama 2 models, you need to login into Hugging Face CLI:

```sh
huggingface-cli login
```

Alternatively, in case you are not using an interactive shell, you can login in non-interactive mode:

```sh
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<YOUR_HUGGING_FACE_TOKEN>')"
```

On TPU Pods:

```sh
./podrun -i -- ~/venv/bin/python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<YOUR_HUGGING_FACE_TOKEN>')"
```

### Convert parameters

```sh
python scripts/convert_params_runner.py llama1-7B
python scripts/convert_params_runner.py llama2-7B
python scripts/convert_params_runner.py llama2-70B
```

### Generation

```sh
python generate.py
```

On TPU pods, the command is:

```sh
./podrun -icw python generate.py
```

### Login into W&B

Before training, you need to login into W&B:

```sh
wandb login <YOUR_WANDB_API_KEY>
```

### Download GSM dataset

I present a simple example of the training pipeline by fine-tuning the model on the GSM dataset.

```sh
cd .. && git clone --depth=1 https://github.com/openai/grade-school-math.git
```

### Training

```sh
python train.py
```

On TPU pods, the command is:

```sh
./podrun -icw python train.py
```

## Model Configurations

- _B_: batch_size
- _L_: seq_len
- _S_: src_seq_len
- _D_: dst_seq_len
- _C_: vocab_size
- _N_: n_layers
- _K_: d_k
- _V_: d_v
- _H_: n_heads_kv
- _R_: n_rep_kv
- _M_: d_model
- _F_: d_ff

| Name | Parameters | _C_ | _N_ | _K_/_V_ | _H_ | _R_ | _M_ | _F_ |
| -: | -: | -: | -: | -: | -: | -: | -: | -: |
| LLaMA 1 7B | 6738415616 | 32000 | 32 | 128 | 32 | 1 | 4096 | 11008 |
| Llama 2 7B | 6738415616 | 32000 | 32 | 128 | 32 | 1 | 4096 | 11008 |
| LLaMA 1 13B |  | 32000 | 40 |  | 40 | 1 | 5120 |  |
| Llama 2 13B | 13015864320 | 32000 | 40 | 128 | 40 | 1 | 5120 | 13824 |
| LLaMA 1 33B |  | 32000 | 60 |  | 52 | 1 | 6656 |  |
| LLaMA 1 65B |  | 32000 | 80 |  | 64 | 1 | 8192 |  |
| Llama 2 70B | 68976648192 | 32000 | 80 | 128 | 8 | 8 | 8192 | 28672 |

```
n_params = 2CM + (2N + 1)M + 2NMRHK + 2NMHK + 3NMF
```

## Model Architecture

<img src="assets/llama.png" width="700px">

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
- There is no bias in the _Q_, _K_, _V_ matrices and the linear projections in the FFNs, which is the same as the original transformer, but different from BERT and BART.
- In Llama models, each FFN has 3 linear projections, while in BART there are only 2.
- There is no dropout in the original LLaMA implementation.
- Llama 2 70B utilises Grouped-Query Attention (GQA).
- Many people fine-tunes Llama in a 16-bit precision (float16 or bfloat16), but the performance would be impacted and thus comparisons with other models trained in 32-bit precision would be unfair. Another thing that is worth noticing is that the parameters for rotary embedding should be always in 32-bit to avoid collision.
