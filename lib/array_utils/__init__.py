import torch
from jax import Array
import jax.numpy as jnp
import numpy as np

def jax2np(x: Array) -> np.ndarray:
    '''
    Converts a JAX array into a NumPy array.

    Args:
        x (Array): JAX array to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    return np.asarray(x)

def np2jax(x: np.ndarray) -> Array:
    '''
    Converts a NumPy array into a JAX array.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        Array: Converted JAX array.
    '''
    return jnp.asarray(x)

def pt2np(x: torch.Tensor) -> np.ndarray:
    '''
    Converts a PyTorch tensor into a NumPy array.

    Args:
        x (torch.Tensor): PyTorch tensor to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    with torch.no_grad():
        return x.numpy()

def np2pt(x: np.ndarray) -> torch.Tensor:
    '''
    Converts a NumPy array into a PyTorch tensor.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        torch.Tensor: Converted PyTorch tensor.
    '''
    return torch.from_numpy(x)

def jax2pt(x: Array) -> torch.Tensor:
    '''
    Converts a JAX array into a PyTorch tensor using NumPy as intermediate.

    Args:
        x (Array): JAX array to convert.

    Returns:
        torch.Tensor: Converted PyTorch tensor.
    '''
    return np2pt(jax2np(x))

def pt2jax(x: torch.Tensor) -> Array:
    '''
    Converts a PyTorch tensor into a JAX array using NumPy as intermediate.

    Args:
        x (torch.Tensor): PyTorch tensor to convert.

    Returns:
        Array: Converted JAX array.
    '''
    return np2jax(pt2np(x))
