from functools import partial
import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
import optax
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable, Optional

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, TrainData, gsm_collate_fn_train
from lib.loss import cross_entropy_loss
from lib.model import Llama, llama_model, model_config_llama2_7B
from lib.multihost_utils import shard_model_params_to_multihost
from lib.param_utils import load_params, save_params
from lib.proc_init_utils import initialise_tpu

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

    lr = 0.00008
    batch_size = 6
    n_accumulation_steps = 8
    max_len = 640
    n_epochs = 7
    seed = 3407

    initialise_tpu('v4-16', n_devices=8, rank=0)
    is_process_0 = jax.process_index() == 0
    cpu_device = jax.devices('cpu')[0]

    if is_process_0:
        import wandb
        wandb.init(project='llama-finetuning-gsm')
        initialise_tracking()

    key = rand.PRNGKey(seed)  # TODO: how to shard this?
    tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/llama2-7B')
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')
    params = shard_model_params_to_multihost(params)
    if is_process_0:
        print('Successfully loaded and sharded model parameters!')

    optimizer = optax.adamw(learning_rate=lr)
    optimizer = optax.MultiSteps(optimizer, n_accumulation_steps)
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    for _ in range(n_epochs):
        pbar = tqdm(total=len(dataloader) // n_accumulation_steps)
        step_loss = 0.0
        total_loss = jnp.zeros(())

        if is_process_0:
            def report_to_wandb(start_time, opt_state, loss):
                nonlocal step_loss
                step_loss += loss.item()
                if optimizer.has_updated(opt_state):
                    wandb.log({'train loss': step_loss / n_accumulation_steps, 'time': time.time() - start_time})
                    step_loss = 0.0
                    pbar.update()

        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            if is_process_0:
                jax.debug.callback(report_to_wandb, start_time, opt_state, loss)

        if is_process_0:
            wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    gathered_params = process_allgather(params)
    if is_process_0:
        save_params(gathered_params, f'{wandb.run.name}.pickle')  # type: ignore

if __name__ == '__main__':
    main()
