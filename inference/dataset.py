import torch
from torch.utils.data import Dataset

class SpotifyGestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data # b 2 c h w
        self.labels = labels # b

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx] # 2 c h w, 1