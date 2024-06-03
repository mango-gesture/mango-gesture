import jax 
import jax.numpy as jnp
from jax import random
from flax import struct as flax_struct
import struct
from typing import Optional


@flax_struct.dataclass
class MLP_config():
    """ Configuration for the MLP """
    name: str
    sizes: list
    modality: str
    classes: int
    # for RGB
    c: Optional[int] = None
    h: Optional[int] = None
    w: Optional[int] = None
    # for JPEG, measured in bytes
    image_size: Optional[int] = None


def save_cfg(cfg, filename):
    if cfg.modality == 'RGB':
        with open(filename, 'w') as file:
            file.write(f'name: {cfg.name}\n')
            file.write(f"modality: RGB\n")
            file.write(f'sizes: {cfg.sizes}\n')
            file.write(f'c: {cfg.c}\n')
            file.write(f'h: {cfg.h}\n')
            file.write(f'w: {cfg.w}\n')
            file.write(f'classes: {cfg.classes}\n')
    else:
        with open(filename, 'w') as file:
            file.write(f'name: {cfg.name}\n')
            file.write(f"modality: JPEG\n")
            file.write(f'sizes: {cfg.sizes}\n')
            file.write(f'image_size: {cfg.image_size}\n')
            file.write(f'classes: {cfg.classes}\n')

def read_cfg(filename):
    c, h, w = None, None, None
    image_size = None
    
    with open(filename, 'r') as file:
        name = file.readline().split(': ')[1].strip()
        modality = file.readline().split(': ')[1].strip()
        sizes = file.readline().split(': ')[1].strip()
        if modality == 'RGB':
            c = int(file.readline().split(': ')[1].strip())
            h = int(file.readline().split(': ')[1].strip())
            w = int(file.readline().split(': ')[1].strip())
        if modality == 'JPEG':
            image_size = int(file.readline().split(': ')[1].strip())
        classes = int(file.readline().split(': ')[1].strip())
    if modality == 'RGB':
        return MLP_config(name, eval(sizes), modality, classes, c, h, w)
    else:
        return MLP_config(name, eval(sizes), modality, classes, image_size=image_size)
        


def initialize_mlp(sizes, key):
    """ Initialize the weights of the MLP """
    scale = 1e-2
    keys = random.split(key, len(sizes))
    weights = []
    for i in range(1, len(sizes)):
        w_key, b_key = random.split(keys[i-1])
        weights.append((random.normal(w_key, (sizes[i-1], sizes[i])) * scale, random.normal(b_key, (sizes[i],)) * scale))
    return weights

# output 3 -> left  | blank | right
def qs_mlp_rgb(c, h, w, sizes, key, output=3): 
    """ Helper function to initialize the MLP for RGB data """
    sizes = [2 * c * h * w] + sizes + [output]
    return initialize_mlp(sizes, key)

def qs_mlp_jpeg(image_size, sizes, key, output=3):
    """ Helper function to initialize the MLP for JPEG data """
    return initialize_mlp([image_size * 2] + sizes + [output], key)

def get_mlp_from_cfg(cfg, key):
    if cfg.modality == 'RGB':
        return qs_mlp_rgb(cfg.c, cfg.h, cfg.w, cfg.sizes, key, cfg.classes)
    else:
        return qs_mlp_jpeg(cfg.image_size, cfg.sizes, key, cfg.classes)

def mlp_forward(weights, x):
    """Forward pass of the MLP """
    for w, b in weights:
        x = x @ w + b
        x = jax.nn.relu(x) # relu on last layer is fine
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

# NOT WORKING, just use the pickle
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