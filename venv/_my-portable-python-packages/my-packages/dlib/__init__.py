
import torch
import numpy as np
from .metrics import *
from .datasets import DatasetSampling


class Metric(object):
    """ Disentanglement metric. """
    def __init__(self,  func_encoder, device="cuda"):
        """
        :param func_encoder: A mapping which takes x as input, and outputs z. (batch, ...) -> (batch, nz)
        :param device:
        """
        @torch.no_grad()
        def _repr_fn(_x):
            _z = func_encoder(torch.from_numpy(_x).to(device)).cpu().numpy()
            return _z

        # Setup encoder.
        self._repr_fn = _repr_fn

    def __call__(self, dataset, *metric):
        if not metric: metric = all_metrics
        # 1. Get dataset
        sampling_dataset = DatasetSampling(dataset)
        # 2. Compute.
        results = {k: eval('compute_%s' % k)(sampling_dataset, self._repr_fn, np.random.RandomState()) for k in metric}
        # Return
        return results
