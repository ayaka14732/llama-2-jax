from flax.serialization import msgpack_serialize, msgpack_restore

def save_params(params, filename):
    serialized_params = msgpack_serialize(params)
    with open(filename, 'wb') as f:
        f.write(serialized_params)
