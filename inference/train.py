import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from mlp import mlp_forward, mlp_serialize_binary, qs_mlp
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--batch_size', type=int, default=32)