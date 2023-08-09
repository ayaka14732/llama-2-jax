from functools import partial
import jax
from jax import Array
import jax.random as rand
import jax_smi
import optax
from transformers import LlamaTokenizer
from typing import Any, Callable

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, TrainData, gsm_collate_fn_train
from lib.llama import Llama, forward_llama, model_config_llama2_7B
from lib.loss import cross_entropy_loss
from lib.proc_init_utils import initialise_tpu
from lib.trainer import trainer

@jax.jit
@jax.value_and_grad
def train_forward_inner(params: Llama, data_batch: TrainData, *, key: rand.KeyArray):
    seq, seq_mask, labels, labels_mask = data_batch
    logits = forward_llama(params, seq, seq_mask, key=key, model_config=model_config_llama2_7B)
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

def train_forward_factory(optimize: Callable, params: Llama, opt_state: Any, step_loss: Any, epoch_loss: Any, data_batch: TrainData, key: rand.KeyArray) -> tuple[Llama, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward_inner(params, data_batch, key=subkey)
    step_loss += loss
    epoch_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, step_loss, epoch_loss, key

def main() -> None:
    lr = 0.0001
    batch_size = 6
    n_accumulation_steps = 8
    max_len = 640
    n_epochs = 7
    seed = 3407
    params_path = 'llama2-7B.pickle'

    initialise_tpu('v3-8')
    jax_smi.initialise_tracking()

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)
    optimizer = optax.adamw(learning_rate=lr)
    train_forward = partial(train_forward_factory, optimizer.update)

    trainer(
        wandb_project_name='llama-finetuning-gsm',
        lr=lr,
        batch_size=batch_size,
        n_accumulation_steps=n_accumulation_steps,
        n_epochs=n_epochs,
        seed=seed,
        params_path=params_path,
        dataloader=dataloader,
        train_forward=train_forward,
        optimizer=optimizer,
    )

if __name__ == '__main__':
    main()
