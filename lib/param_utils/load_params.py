import pickle
from typing import Any

def load_params_bytes(b: bytes) -> Any:
    return pickle.loads(b)

def load_params(filename: str) -> Any:
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params
