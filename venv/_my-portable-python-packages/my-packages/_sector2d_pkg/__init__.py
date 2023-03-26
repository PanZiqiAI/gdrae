
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Sector2DUnif(Dataset):
    """
    A 2-dimensional sector dataset. (unif)
    """
    def __init__(self, n_samples=10000, device='cuda'):
        # Config.
        self._device = device
        """ Load model. """
        from ._sector2d_trainer.config import ConfigTrainModel
        from ._sector2d_trainer.f_trainer import Trainer
        # 1. Generate config.
        cfg = ConfigTrainModel(load_rel_path=("default", 1678, 99999), block_argv=True, deploy=False)
        # 2. Generate model
        model = Trainer(cfg=cfg)
        """ Set F. """
        self._F = model.f.to(device)
        """ Set data. """
        # 1. Get z_rand.
        z_rand_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'z_rand_files')
        z_rand_path = os.path.join(z_rand_dir, 'n_samples=%d.npy' % n_samples)
        # (1) Generate & save to file.
        if not os.path.exists(z_rand_path):
            z_rand = torch.rand(size=(n_samples, 2)).numpy()
            if not os.path.exists(z_rand_dir): os.makedirs(z_rand_dir)
            np.save(z_rand_path, z_rand)
        # (2) Load from file.
        else:
            z_rand = np.load(z_rand_path)
        # 2. Get np dataset. (n_samples, 2)
        self._data = self.convert(z=torch.from_numpy(z_rand).to(device=device)).cpu().numpy()

    @property
    def data(self):
        return self._data

    def convert(self, z):
        radius, angles = z.chunk(2, dim=1)
        radius, angles = self._F(radius)+1.0, angles*(np.pi/2.0)
        data_x, data_y = radius*torch.cos(angles), radius*torch.sin(angles)
        # Return
        return torch.cat([data_x, data_y], dim=1)

    def sampling(self, n_samples):
        z = torch.rand(size=(n_samples, 2), device=self._device)
        return self.convert(z)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


__all__ = ['Sector2DUnif']
