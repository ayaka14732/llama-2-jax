from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
import optax
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable, Optional
import wandb

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, TrainData, gsm_collate_fn_train
from lib.loss import cross_entropy_loss
from lib.model import Llama, llama_model, model_config_llama2_7B
from lib.param_utils import load_params, save_params
# from lib.proc_init_utils import initialise_gpu
from lib.proc_init_utils import initialise_tpu

from lib.multihost_utils import shard_array_to_multihost
from lib.tree_utils import tree_apply
from lib.model import Llama, LlamaModel, Decoder, Attention

optimize: Optional[Callable]

@jax.value_and_grad
def train_forward(params: Llama, data_batch: TrainData, *, key: rand.KeyArray):
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model(params.model, seq, seq_mask, key=key, model_config=model_config_llama2_7B)
    logits = outputs @ params.lm_head
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@jax.jit
def train_step(params: dict, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray) -> tuple[dict, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def main() -> None:
    global optimize

    lr = 2e-5
    batch_size = 6
    n_gradient_accumulation_steps = 8
    max_len = 640
    n_epochs = 4
    seed = 3407

    # initialise_gpu(cuda_visible_devices='0,1,2,3')
    initialise_tpu('v4-16', n_devices=8, rank=0)
    is_process_0 = jax.process_index() == 0
    if is_process_0:
        wandb.init(project='llama-finetuning-gsm')
        initialise_tracking()

    key = rand.PRNGKey(seed)
    tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/llama2-7B')
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    cpu_device = jax.devices('cpu')[0]
    default_devices = jax.devices()

    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')

    sharding = Llama(
        model=LlamaModel(
            embedding=...,
            decoder=Decoder(
                input_norm=...,
                attention=Attention(
                    q_proj=3,
                    k_proj=2,
                    v_proj=2,
                    out_proj=2,
                ),
                post_attn_norm=...,
                gate_proj=2,
                up_proj=2,
                down_proj=1,
            ),
            norm=...,
        ),
        lm_head=...,
    )
    params = tree_apply(shard_array_to_multihost, params, sharding)
    if is_process_0:
        print('Successfully loaded and sharded model parameters!')

    shard_all = lambda x: jax.tree_map(lambda a: shard_array_to_multihost(a, ...), x)

    optimizer = optax.chain(
        optax.adamw(learning_rate=lr),
        optax.apply_every(n_gradient_accumulation_steps),
    )
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    for _ in range(n_epochs):
        pbar = tqdm(total=len(dataloader))
        total_loss = jnp.zeros(())
        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            data_batch = shard_all(data_batch)
            # TODO: save model
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            if is_process_0:
                jax.debug.callback(lambda loss: wandb.log({'train loss': loss.item(), 'time': time.time() - start_time}) or pbar.update(), loss)
        if is_process_0:
            wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    if is_process_0:
        save_params(params, f'{wandb.run.name}.pickle')  # type: ignore

if __name__ == '__main__':
    main()
