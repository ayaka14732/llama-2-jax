from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
import jax_smi
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
from lib.proc_init_utils import initialise_gpu

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
    batch_size = 2
    max_len = 640
    n_epochs = 3
    seed = 3407

    initialise_gpu(cuda_visible_devices='0')
    # initialise_tpu('v4-16', n_devices=1, rank=0)
    jax.experimental.compilation_cache.compilation_cache.initialize_cache('cache')
    wandb.init(project='llama-finetuning-gsm')
    jax_smi.initialise_tracking()
    key = rand.PRNGKey(seed)

    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    params = load_params('llama2-7B.pickle')

    optimizer = optax.adamw(learning_rate=lr)
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    for _ in range(n_epochs):
        pbar = tqdm(total=len(dataloader))
        total_loss = jnp.zeros(())
        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            # TODO: save model
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            jax.debug.callback(lambda loss: wandb.log({'train loss': loss.item(), 'time': time.time() - start_time}) or pbar.update(), loss)
        wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    save_params(params, f'{wandb.run.name}.pickle')  # type: ignore

if __name__ == '__main__':
    main()
