import torch
from torch.utils.data import Dataset

class SineWaveTorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]