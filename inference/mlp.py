import jax 
import jax.numpy as jnp
from jax import random
import struct

def initialize_mlp(sizes, key):
    """ Initialize the weights of the MLP """
    scale = 1e-2
    keys = random.split(key, len(sizes))
    weights = []
    for i in range(1, len(sizes)):
        w_key, b_key = random.split(keys[i-1])
        weights.append((random.normal(w_key, (sizes[i-1], sizes[i])) * scale, random.normal(b_key, (sizes[i],)) * scale))
    return weights

def qs_mlp(c, h, w, sizes, key, output=1):
    """ Helper function to initialize the MLP """
    sizes = [2 * c * h * w] + sizes + [output]
    return initialize_mlp(sizes, key)

def mlp_forward(weights, x):
    """Forward pass of the MLP """
    for w, b in weights:
        x = x @ w + b
        x = jax.nn.relu(x)
        # x = jax.nn.gelu(x) * (b.shape[0] > 1) + jax.nn.softplus(x) * (b.shape[0] == 1) # final layer is positive
    return x

batch_mlp_forward = jax.vmap(mlp_forward, in_axes=(None, 0), out_axes=0)

def mlp_serialize_binary(params, filename):
    weights = [w for w, _ in params]
    biases = [b for _, b in params]

    with open(filename, 'wb') as file:
        # Write the number of layers
        num_layers = len(weights)
        file.write(struct.pack('I', num_layers))  # Unsigned integer

        # Write each layer's size
        for w in weights:
            neurons_in = w.shape[0]
            neurons_out = w.shape[1]
            file.write(struct.pack('II', neurons_in, neurons_out))

        # Write the weights and biases
        for w, b in zip(weights, biases):
            w = jnp.array(w).astype(jnp.float32)
            b = jnp.array(b).astype(jnp.float32)

            file.write(w.tobytes())
            file.write(b.tobytes())

# NOT WORKING
def mlp_deserialize_binary(filename):
    with open(filename, 'rb') as file:
        # Read the number of layers
        num_layers = struct.unpack('I', file.read(4))[0]

        # Read each layer's size
        sizes = []
        for _ in range(num_layers):
            neurons_in, neurons_out = struct.unpack('II', file.read(8))
            sizes.append((neurons_in, neurons_out))

        # Read the weights and biases
        weights = []
        biases = []
        for (neurons_in, neurons_out) in sizes:
            w = struct.unpack(f'{neurons_in * neurons_out}f', file.read(4 * neurons_in * neurons_out))
            b = struct.unpack(f'{neurons_out}f', file.read(4 * neurons_out))

            weights.append(jnp.array(w).reshape(neurons_in, neurons_out))
            biases.append(jnp.array(b))

    return list(zip(weights, biases))