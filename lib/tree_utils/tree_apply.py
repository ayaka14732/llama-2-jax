import jax

def tree_apply(func, *pytrees):
    '''
    ```python
    >>> tree_apply(lambda a, b: a + b + 1, [[[1, 2]]], [[[3, 4]]])
    [[[5, 7]]]
    ```
    '''
    leaves, treedefs = zip(*([jax.tree_util.tree_flatten(pytree) for pytree in pytrees]), strict=True)
    treedef = treedefs[0]
    results = [func(*args) for args in zip(*leaves, strict=True)]
    return treedef.unflatten(results)
