from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from lib.tree_utils import tree_apply

assert tree_apply(lambda a, b: a + b + 1, [[[1, 2]]], [[[3, 4]]]) == [[[5, 7]]]

def f(a, b):
    if a is ...:
        return b
    if b is ...:
        return a
    return a + b + 1

assert tree_apply(f, [[[1, 2, 3, ...]]], [[[4, 5, ..., 6]]]) == [[[6, 8, 3, 6]]]
