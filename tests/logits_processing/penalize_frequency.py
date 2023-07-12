from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import jax.random as rand

batch_size = 1
vocab_size = 5
key = rand.PRNGKey(42)
logits = rand.normal(key, (batch_size, vocab_size), dtype=jnp.float_)
seq = jnp.array([[1, 2, 3, 4], [2, 3, 3, 0]], dtype=jnp.uint16)
attn_mask = jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.bool_)

# logits
# presence_penalty(logits, seq, attn_mask, 0.8)
# frequency_penalty(logits, seq, attn_mask, 0.8)
