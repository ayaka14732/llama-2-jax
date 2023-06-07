import torch
from jax import Array
import jax.numpy as jnp
import numpy as np

def jax2np(x: Array) -> np.ndarray:
    return np.asarray(x)

def np2jax(x: np.ndarray) -> Array:
    return jnp.asarray(x)

def pt2np(x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return x.numpy()

def np2pt(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)

def jax2pt(x: Array) -> torch.Tensor:
    return np2pt(jax2np(x))

def pt2jax(x: torch.Tensor) -> Array:
    return np2jax(pt2np(x))
