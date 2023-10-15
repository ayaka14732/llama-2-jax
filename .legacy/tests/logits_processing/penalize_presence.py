from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp

from lib.logits_processing import penalize_presence

batch_size = 2
seq_len = 4
vocab_size = 5

logits = jnp.array([[0., 1., 2., 3., 4.], [1., 2., 3., 4., 5.]])
seq = jnp.array([[1, 2, 3, 4], [2, 3, 3, 0]], dtype=jnp.uint16)
attn_mask = jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.bool_)
# seq1: 1, 2, 3
# seq2: 0, 2, 3

assert logits.shape == (batch_size, vocab_size)
assert seq.shape == (batch_size, seq_len)
assert attn_mask.shape == (batch_size, seq_len)

y = jnp.array([[0., 0., 1., 2., 4.], [0., 2., 2., 3., 5.]])
y_hat = penalize_presence(logits, seq, attn_mask, 1.)
assert jnp.array_equal(y, y_hat)
