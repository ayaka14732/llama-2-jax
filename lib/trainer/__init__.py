from functools import partial
import jax
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand
import signal
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, NamedTuple

from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.llama import Llama

def _load_params_from_disk(path: str) -> Llama:
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params(path)
    params = shard_model_params(params)
    return params

def _init_wandb_project(wandb_project_name: str, wandb_config: dict, is_process_0: bool) -> Any:
    if not is_process_0:
        return None
    else:
        import wandb
        wandb.init(project=wandb_project_name, config=wandb_config)
        return wandb

def _fori_loop(lower: int, upper: int, body_fun: Callable, initial_state: Any) -> Any:
    state = initial_state
    for i in range(lower, upper):
        state = body_fun(i, state)
    return state

def _set_signal_handler(handler: Any) -> None:
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

def _save_params_to_disk(params: Llama, is_process_0: bool) -> None:
    _set_signal_handler(signal.SIG_IGN)
    gathered_params = process_allgather(params)
    if is_process_0:
        save_params(gathered_params, f'{wandb.run.name}.pickle')  # type: ignore
    _set_signal_handler(_signal_handler)

def _signal_handler(signum, frame):
    print(f'Signal {signum} received. Saving params ...')
    _save_params_to_disk()
    exit(0)

class _TrainState(NamedTuple):
    key: rand.KeyArray
    params: Llama
    opt_state: Any  # TODO: typing this

def _report_to_wandb_factory(wandb, n_accumulation_steps, pbar, is_process_0, start_time, step_loss):
    if is_process_0:
        # if optimizer.has_updated(opt_state):
        step_loss_ = step_loss.item() / n_accumulation_steps
        wandb.log({'step loss': step_loss_, 'time': time.time() - start_time})
        step_loss = 0.0
        pbar.update()

def _train_step_factory(train_forward: Callable, dataloader: DataLoader, n_accumulation_steps: Any, is_process_0: bool, wandb, i: int, state: _TrainState) -> _TrainState:  # TODO: better function name
    key, params, opt_state = state
    n_steps = len(dataloader)

    pbar = tqdm(total=len(dataloader) // n_accumulation_steps)
    report_to_wandb = partial(_report_to_wandb_factory, wandb, n_accumulation_steps, pbar, is_process_0)
    step_loss = jnp.zeros(())
    epoch_loss = jnp.zeros(())

    for data_batch in dataloader:
        start_time = time.time()
        params, opt_state, step_loss, epoch_loss, key = train_forward(params, opt_state, step_loss, epoch_loss, data_batch, key)
        jax.debug.callback(report_to_wandb, start_time, step_loss)

    if is_process_0:
        epoch_loss_ = epoch_loss.item() / n_steps
        wandb.log({'epoch loss': epoch_loss_}, commit=False)

    return _TrainState(key, params, opt_state)

def trainer(
    wandb_project_name: str,
    lr: float,
    batch_size: int,
    n_accumulation_steps: int,
    n_epochs: int,
    seed: int,
    params_path: str,
    dataloader: DataLoader,
    train_forward: Callable,
    optimizer: Any,
):
    is_process_0 = jax.process_index() == 0
    wandb_config = {
        'Learning rate': lr,
        'Batch size': batch_size * n_accumulation_steps,
        'Epochs': n_epochs,
        'Optimiser': 'adamw',  # TODO: Change this
    }
    wandb = _init_wandb_project(wandb_project_name, wandb_config, is_process_0)
    params = _load_params_from_disk(params_path)
    _set_signal_handler(_signal_handler)

    key = rand.PRNGKey(seed)
    opt_state = optimizer.init(params)
    initial_state = _TrainState(key, params, opt_state)
    train_step = partial(_train_step_factory, train_forward, dataloader, n_accumulation_steps, is_process_0, wandb)
    _fori_loop(0, n_epochs, train_step, initial_state)
