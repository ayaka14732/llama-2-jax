from lib.proc_init_utils import initialise_tpu; initialise_tpu('v3-32')

import einops as op
from functools import partial
import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand
import jax_smi
import math
import optax
import signal
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable
import wandb

from lib.data import TrainData
from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, gsm_collate_fn_train
from lib.LLM import Llama, RotaryValues, forward_llama, init_llama, make_rotary_values
# from lib.LLM import model_config_dummy as model_config
from lib.LLM import model_config_llama2_7B as model_config
from lib.loss import cross_entropy_loss
from lib.multihost_utils import shard_model_params
from lib.param_utils import load_params, save_params

is_process_0: bool
params: Llama
optimize: Callable

def load_params_from_disk(path: str) -> Llama:
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        # params = init_llama(key=rand.key(42), model_config=model_config)
        params = load_params(path)
    params = shard_model_params(params)
    return params

def set_save_params_signal():
    signal.signal(signal.SIGINT, save_params_signal_handler)
    signal.signal(signal.SIGTERM, save_params_signal_handler)

def unset_save_params_signal():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

def save_params_to_disk() -> None:
    unset_save_params_signal()
    gathered_params = process_allgather(params)
    if is_process_0:
        save_params(gathered_params, f'{wandb.run.name}.pickle')  # type: ignore
    set_save_params_signal()

def save_params_signal_handler(signum, frame):
    save_params_to_disk()
    print(f'Signal {signum} received. Model params have been successfully saved to disk.')
    exit(-1)

@jax.value_and_grad
def train_forward(params: Llama, rotary_values: RotaryValues, data_batch: TrainData, *, key: Array):
    seq, seq_mask, labels, labels_mask = data_batch
    qk_mask = op.rearrange(jnp.tril(op.einsum(seq_mask, seq_mask, 'B L1, B L2 -> B L1 L2')), 'B L1 L2 -> B 1 1 L1 L2')  # causal QK mask
    logits, _ = forward_llama(params, seq, qk_mask, rotary_values=rotary_values, key=key, model_config=model_config)
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@jax.jit
def train_step(params: Llama, opt_state: Any, rotary_values: RotaryValues, total_loss: Array, data_batch: TrainData, key: Array) -> tuple[Llama, Any, Array, Array, Array]:
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, rotary_values, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def main() -> None:
    global is_process_0, params, optimize

    lr = 0.00005
    batch_size = 6
    n_accumulation_steps = 8
    max_len = 640
    n_epochs = 7
    seed = 3407

    jax.distributed.initialize()
    jax_smi.initialise_tracking()
    is_process_0 = jax.process_index() == 0

    if is_process_0:
        wandb.init(project='llama-finetuning-gsm', config=dict(learning_rate=lr, batch_size=batch_size * n_accumulation_steps, n_epochs=n_epochs, optimiser='adamw'))

    key = rand.key(seed, impl='rbg')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed, drop_last=True)  # TODO: setting `drop_last` because the `batch_size` of `rotary_values` is not properly handled

    params = load_params_from_disk('llama2-7B.pickle')
    set_save_params_signal()

    n_steps = math.ceil(len(dataloader) / n_accumulation_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=lr,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=lr,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, n_accumulation_steps)
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    rotary_values = make_rotary_values(None, batch_size, max_len, model_config=model_config)

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
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, rotary_values, total_loss, data_batch, key)
            if is_process_0:
                jax.debug.callback(report_to_wandb, start_time, opt_state, loss)

        if is_process_0:
            wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    save_params_to_disk()

if __name__ == '__main__':
    main()
