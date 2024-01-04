import jax
device_count = jax.device_count()
local_device_count = jax.local_device_count()
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

if jax.process_index() == 0:
    print('global device count:', jax.device_count())
    print('local device count:', jax.local_device_count())
    print('pmap result:', r)
