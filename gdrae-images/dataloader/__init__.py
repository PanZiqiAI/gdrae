
from torch.utils.data.dataloader import DataLoader
from custom_pkg.pytorch.operations import DataCycle
from disentangling.datasets import Shapes, Faces, Shapes3D, Cars3D, SmallNorb


def generate_data(cfg):
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Get dataset.
    # ------------------------------------------------------------------------------------------------------------------
    if cfg.args.dataset == 'shapes2d':
        dataset = Shapes()
    elif cfg.args.dataset == 'faces':
        dataset = Faces()
    elif cfg.args.dataset == 'shapes3d':
        dataset = Shapes3D()
    elif cfg.args.dataset == 'cars3d':
        dataset = Cars3D()
    elif cfg.args.dataset == 'small_norb':
        dataset = SmallNorb()
    else:
        raise ValueError
    # ------------------------------------------------------------------------------------------------------------------
    # 2. Load dataset.
    # ------------------------------------------------------------------------------------------------------------------
    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.args.batch_size, drop_last=cfg.args.dataset_drop_last,
        shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)
    # Return
    return DataCycle(dataloader) 
