from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
import jax_smi
import optax
import time
from transformers import LlamaTokenizer
from typing import Any, Callable, Optional
import wandb

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, TrainData, train_collate_fn_factory
from lib.loss import cross_entropy_loss
from lib.model import Llama, config_7B, llama_model
from lib.param_utils import load_params, save_params
from lib.proc_init_utils import initialise_gpu

optimize: Optional[Callable]

@jax.value_and_grad
def train_forward(params: Llama, data_batch: TrainData, *, key: rand.KeyArray):  # TODO: dropout
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model(params.model, seq, seq_mask, config=config_7B)
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

    lr = 0.002
    batch_size = 16
    max_len = 512
    n_epochs = 8
    seed = 3407

    initialise_gpu(cuda_visible_devices='2')  # 0,1,2,3
    wandb.init(project='llama-finetuning-gsm')
    print(wandb.run.name)  # type: ignore
    jax_smi.initialise_tracking()
    key = rand.PRNGKey(seed)

    tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/7B')

    dataset = GSMDataset(split='train')
    tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/7B')
    collate_fn = partial(train_collate_fn_factory, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    params = load_params('7B.pickle')

    optimizer = optax.adafactor(learning_rate=lr)
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    for _ in range(n_epochs):
        total_loss = jnp.zeros(())
        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            jax.debug.callback(lambda loss: wandb.log({'train loss': loss.item(), 'time': time.time() - start_time}), loss)
        wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    save_params(params, f'{wandb.run.name}.npy')  # type: ignore

if __name__ == '__main__':
    main()
