# TODO: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html
def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
