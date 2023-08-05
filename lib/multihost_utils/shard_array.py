import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from types import EllipsisType

def shard_array(arr: Array, axis: int | EllipsisType) -> Array:
    shape = arr.shape
    devices: np.ndarray = np.array(jax.devices())

    if axis is ...:
        mesh = Mesh(devices, ('a',))
        sharding = NamedSharding(mesh, P(None))
    else:
        sharding_tuple_ = [1] * len(shape)
        sharding_tuple_[axis] = -1
        sharding_tuple = tuple(sharding_tuple_)
        name_tuple = tuple('abcdefghijklmnopqrstuvwxyz'[:len(shape)])

        mesh = Mesh(devices.reshape(sharding_tuple), name_tuple)
        sharding = NamedSharding(mesh, P(*name_tuple))

    xs = [jax.device_put(arr[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, xs)
