
import torch
import numpy as np
from _sector2d_pkg import Sector2DUnif
from torch.utils.data import Dataset, DataLoader
from custom_pkg.pytorch.operations import DataCycle


class Sector2D(Dataset):
    """
    A 2-dimensional sector dataset.
    """
    def __init__(self, n_samples=10000):
        # Generate np dataset. (n_samples, 2)
        self._data = self.convert(z=torch.rand(size=(n_samples, 2))).numpy()

    @property
    def data(self):
        return self._data

    @staticmethod
    def convert(z):
        radius, angles = z.chunk(2, dim=1)
        radius, angles = radius + 1.0, angles * (np.pi/2.0)
        data_x, data_y = radius*torch.cos(angles), radius*torch.sin(angles)
        # Return
        return torch.cat([data_x, data_y], dim=1)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def generate_data(cfg):
    # 1. Get dataset.
    if cfg.args.dataset == 'sector2d': dataset = Sector2D(n_samples=cfg.args.n_samples)
    elif cfg.args.dataset == 'sector2dunif': dataset = Sector2DUnif(n_samples=cfg.args.n_samples, device=cfg.args.device)
    else: raise ValueError
    # 2. Get dataloader.
    dataloader = DataLoader(
        dataset, batch_size=cfg.args.batch_size, drop_last=cfg.args.dataset_drop_last,
        shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)
    # Return
    return DataCycle(dataloader)
