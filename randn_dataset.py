from torch.utils.data import Dataset
import torch


class RandnDataset(Dataset):
    def __init__(self, num_samples=10000):
        super().__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(784)
        return dict(image=image)