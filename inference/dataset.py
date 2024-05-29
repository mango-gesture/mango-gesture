import torch
from torch.utils.data import Dataset
import os
import jax.numpy as jnp
from PIL import Image

NULL_TOKEN = 0xFF
class SpotifyGestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data # b 2 c h w
        self.labels = labels # b

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx] # 2 c h w, 1
    
def get_rgb_dataset(data_path, classes, c, h, w):
    """
    data_path must have subfolders 0, 1, 2, ..., classes-1
    """
    data = []
    labels = []
    for i in range(classes):
        folder = os.path.join(data_path, str(i))
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                img = Image.open(os.path.join(folder, file))
                img = jnp.array(img).astype(jnp.float32)
                #ensure correct shape
                if img.shape[0] != c or img.shape[1] != h or img.shape[2] != w:
                    raise ValueError(f"Image {file} has wrong shape {img.shape}, needed {c} {h} {w}")
                data.append(img)
                labels.append(i)

        else:
            raise ValueError(f"Folder {folder} not found")
        
    data = jnp.stack(data)
    labels = jnp.array(labels)
    return SpotifyGestureDataset(data, labels)

def get_jpeg_dataset(data_path, classes, image_size):
    """
    data_path must have subfolders 0, 1, 2, ..., classes-1
    """
    data = []
    labels = []
    for i in range(classes):
        folder = os.path.join(data_path, str(i))
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                img = Image.open(os.path.join(folder, file))
                img = jnp.array(img).astype(jnp.float32)
                #if the image is less than image_size bytes, append NULL_TOKEN
                if img.shape[0] < image_size:
                    img = jnp.concatenate([img, jnp.array([NULL_TOKEN] * (image_size - img.shape[0]))])
                img = img.resize((image_size, image_size))
                data.append(img)
                labels.append(i)

        else:
            raise ValueError(f"Folder {folder} not found")
        
    data = jnp.stack(data)
    labels = jnp.array(labels)
    return SpotifyGestureDataset(data, labels)