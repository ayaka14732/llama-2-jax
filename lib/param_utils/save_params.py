import pickle
from typing import Any

def save_params_bytes(params: Any) -> bytes:
    return pickle.dumps(params)

def save_params(params: Any, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(params, file=f)
