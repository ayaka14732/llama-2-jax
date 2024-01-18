# LLMs in JAX - Detailed Setup Guide

## Introduction

This repository is an extension of the remarkable project [ayeka/llama-2-jax](https://github.com/ayaka14732/llama-2-jax) by Ayeka. It provides support for a diverse range of Models, including Mistral, Vicuna, Orca, Zephyr, and more. The conversion for using these models through JAX has been successfully implemented for various instances.

The models currently converted include:

- Mistral-7B-v0.1
- Mistral-7B-Instruct-v0.2
- SOLAR-10.7B-v1.0
- SOLAR-10.7B-Instruct-v1.0
- StableBeluga-13B
- StableBeluga-7B
- StableLM-Zephyr-3B
- Zephyr-7B-beta
- llama1-30B
- Vicuna-7B
- Vicuna-13B
- llama1-13B
- llama2-13B
- llama1-7B
- llama2-7B
- Orca-2-13B
- Orca-2-7B
- Mistral-7B-Instruct-v0.1
- llama2-7B-chat
- llama2-13B-chat

You can find converted model pickle files on Hugging Face [here](https://huggingface.co/divyapatel4).

The project is backed by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/).

## Documentation

Comprehensive documentation for this library, authored by Ayeka, is available on [GitHub Pages](https://ayaka14732.github.io/llama-2-jax/).

## Environment Setup

Run this script on TPU-VMs for setting-up environment.
```
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11-full -y
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install jupyter notebook
git clone https://github.com/divyapatel4/llama-2-jax.git
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/deepmind/optax.git  # https://github.com/google-deepmind/optax/issues/472
pip install -r llama-2-jax/requirements.txt
sudo apt-get install libpython3.11
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('TOKEN')"

```
### Special Configuration for TPU Pods

If using TPU pods, populate `~/podips.txt` with the IP addresses of all hosts. Ensure local SSH access to itself and all IPs listed.


On TPU Pods:

```sh
./podrun -i -- ~/venv/bin/pip install git+https://github.com/huggingface/transformers.git
./podrun -i -- ~/venv/bin/pip install git+https://github.com/deepmind/optax.git
./podrun -iw -- ~/venv/bin/pip install -r requirements.txt
```

### Login into Hugging Face CLI

If working with Llama 2 models, log in to Hugging Face CLI:

```sh
huggingface-cli login
```

In non-interactive mode:
```sh
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<YOUR_HUGGING_FACE_TOKEN>')"
```

On TPU Pods:

```sh
./podrun -i -- ~/venv/bin/python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<YOUR_HUGGING_FACE_TOKEN>')"
```

### Convert Parameters
```sh
python scripts/convert_params_runner.py llama2-7B
```

### Generation
```sh
python generate.py
```

On TPU Pods:
```sh
./podrun -icw ~/venv/bin/python generate.py
```

### Login into W&B

Before training, log in to W&B:

```sh
wandb login <YOUR_WANDB_API_KEY>
```

### Download GSM Dataset

For a training example, clone the Grade School Math (GSM) dataset:

```sh
cd .. && git clone --depth=1 https://github.com/openai/grade-school-math.git
```

### Training

```sh
python train.py
```

On TPU Pods:

```sh
./podrun -icw ~/venv/bin/python train.py
```
