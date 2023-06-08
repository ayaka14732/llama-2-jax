from flax.serialization import msgpack_restore

def load_params(filename):
    with open(filename, 'rb') as f:
        serialized_params = f.read()
    return msgpack_restore(serialized_params)
