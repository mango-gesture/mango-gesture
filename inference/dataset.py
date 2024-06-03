import torch
from torch.utils.data import Dataset
import os
import jax.numpy as jnp
from PIL import Image
import jax
import numpy as np
from inference.mlp import MLP_config

NULL_TOKEN = 2.0


class SpotifyGestureDataset(Dataset):
    def __init__(self, data, labels, classes: int):
        self.data = data # b 2 c h w
        self.labels = labels # b
        self.classes = classes
        print(f"Dataset has {len(self.data)} samples")
        print(f"Dataset has {self.classes} classes")
        for i in range(self.classes):
            print(f"Class {i} has {np.sum(self.labels == i)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].ravel(), self.labels[idx] # 2chw, classes
    
# DON'T USE THIS 
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
                img = np.array(img).astype(np.float32)
                #ensure correct shape
                if img.shape[0] != c or img.shape[1] != h or img.shape[2] != w:
                    raise ValueError(f"Image {file} has wrong shape {img.shape}, needed {c} {h} {w}")
                data.append(img)
                labels.append(i)

        else:
            raise ValueError(f"Folder {folder} not found")
        
    data = np.stack(data)
    labels = np.array(labels)
    return SpotifyGestureDataset(data, labels, classes)

def load_jpeg_image(image_path, image_size):
    with open(image_path, 'rb') as file:
        img = np.array(list(file.read())).astype(np.int32)
    img = np.array(img).astype(np.float32) / (255/2) - 1
    if img.shape[0] < image_size:
        img = np.concatenate([img, np.full((image_size - img.shape[0],), NULL_TOKEN)])
    return img

def get_jpeg_dataset(data_path, classes, image_size):
    """
    data_path must have subfolders 0, 1, 2, ..., classes-1
    """
    data = []
    labels = []
    for i in range(classes):
        folder = os.path.join(data_path, str(i))
        if os.path.isdir(folder):
            # files 0 and 1 are a pair, 2 and 3 are a pair, etc
            num_files = len(os.listdir(folder))

            if num_files % 2 != 0:
                print(f"Folder {folder} has an odd number of files")
            
            for j in range(0, num_files, 2):
                if not os.path.exists(os.path.join(folder, "output"+str(j)+".jpg") or not os.path.exists(os.path.join(folder, "output"+str(j+1)+".jpg"))):
                    print(f"Files {os.path.join(folder, 'output'+str(j)+'.jpg')} and {os.path.join(folder, 'output'+str(j+1)+'.jpg')} not found")
                    continue
                # Read img1 as bytes and convert them to ints
                with open(os.path.join(folder, "output"+str(j)+".jpg"), 'rb') as file:
                    img1 = np.array(list(file.read())).astype(np.int32)

                with open(os.path.join(folder, "output"+str(j+1)+".jpg"), 'rb') as file:
                    img2 = np.array(list(file.read())).astype(np.int32)

                # Normalize
                img1 = np.array(img1).astype(np.float32) / (255/2) - 1
                img2 = np.array(img2).astype(np.float32) / (255/2) - 1

                # Append NULL_TOKEN to both images until they are image_size
                # throws away images that are >= image_size, technically waste but it's fine
                if img1.shape[0] < image_size:
                    img1 = np.concatenate([img1, np.full((image_size - img1.shape[0],), NULL_TOKEN)])
                else: 
                    continue
                if img2.shape[0] < image_size:
                    img2 = np.concatenate([img2, np.full((image_size - img2.shape[0],), NULL_TOKEN)])
                else:
                    continue

                data.append(np.concatenate([img1, img2]))
                labels.append(i)
                
                # we can also use the reversed pair for the other label
                data.append(np.concatenate([img2, img1]))
                labels.append(not i) # only works while our labels are 0 or 1
                if i > 1:
                    raise ValueError("Labels are not 0 or 1")

        else:
            raise ValueError(f"Folder {folder} not found")
        
    data = np.stack(data)
    labels = np.array(labels)
    return SpotifyGestureDataset(data, labels, classes)

def get_dataset_from_cfg(data_path, cfg: MLP_config):
    if cfg.modality == 'RGB':
        return get_rgb_dataset(data_path, cfg.classes, cfg.c, cfg.h, cfg.w)
    else:
        return get_jpeg_dataset(data_path, cfg.classes, cfg.image_size)