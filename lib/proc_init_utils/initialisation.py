import os
from typing import Optional

def _find_free_port() -> int:
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def _post_init_general() -> None:
    # post-init flags
    import jax
    jax.config.update('jax_enable_custom_prng', True)
    jax.config.update('jax_default_prng_impl', 'rbg')
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    jax.config.update('jax_spmd_mode', 'allow_all')

def initialise_cpu(n_devices: int=1) -> None:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=' + str(n_devices)

    _post_init_general()

def initialise_gpu(cuda_visible_devices: Optional[str]=None) -> None:
    os.environ['JAX_PLATFORMS'] = ''

    if cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    _post_init_general()

def _initialise_tpu_one_chip(rank: int) -> None:
    port = _find_free_port()
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    if rank not in (0, 1, 2, 3):
        raise ValueError('Rank must be within 0-3.')
    os.environ['TPU_VISIBLE_DEVICES'] = str(rank)
    os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
    os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

def _initialise_tpu_two_chip(rank: int) -> None:
    port = _find_free_port()
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    if rank not in (0, 1):
        raise ValueError('Rank must be either 0 or 1.')
    os.environ['TPU_VISIBLE_DEVICES'] = ('0,1', '2,3')[rank]
    os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
    os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

def _initialise_tpu_four_chip(rank: int) -> None:
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    if rank != 0:
        raise ValueError('Rank must be 0.')
    os.environ['TPU_VISIBLE_DEVICES'] = '0,1,2,3'

def _initialise_tpu_full(rank: int) -> None:
    if rank != 0:
        raise ValueError('Rank must be 0.')

def initialise_tpu(accelerator_type: str, n_devices: int | None=None, rank: int=0) -> None:
    os.environ['JAX_PLATFORMS'] = ''

    if accelerator_type == 'v3-8':
        if n_devices == 2: _initialise_tpu_one_chip(rank)
        elif n_devices == 4: _initialise_tpu_two_chip(rank)
        elif n_devices == 8 or n_devices is None: _initialise_tpu_full(rank)
        else:
            raise ValueError(f'Invalid value `n_devices`: {n_devices}')
    elif accelerator_type in ('v3-32', 'v3-256'):
        if n_devices == 2: _initialise_tpu_one_chip(rank)
        elif n_devices == 4: _initialise_tpu_two_chip(rank)
        elif n_devices == 8: _initialise_tpu_four_chip(rank)
        elif n_devices is None: _initialise_tpu_full(rank)
        else:
            raise ValueError(f'Invalid value `n_devices`: {n_devices}')
    elif accelerator_type == 'v4-16':
        if n_devices == 1: _initialise_tpu_one_chip(rank)
        elif n_devices == 2: _initialise_tpu_two_chip(rank)
        elif n_devices == 4: _initialise_tpu_four_chip(rank)
        elif n_devices == 8 or n_devices is None: _initialise_tpu_full(rank)
        else:
            raise ValueError(f'Invalid value `n_devices`: {n_devices}')
    else:
        raise NotImplementedError('Only the initialisation on Cloud TPU v3-8 and v4-16 is supported.')

    _post_init_general()
