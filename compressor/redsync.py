import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank

class RedSyncCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'redsync'

    @staticmethod
    def clear():
        RedSyncCompressor.residuals = {}
        RedSyncCompressor.sparsities = []
        RedSyncCompressor.zero_conditions = {}
        RedSyncCompressor.values = {} 
        RedSyncCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RedSyncCompressor.residuals[name].data)

            l = 0.0
            r = 1.0
            thres = 0.0
            eps = 0.2
            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)

            while r - l > eps:
                tmp_ratio = l + (r-l)/2
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                nnz = indexes.numel()
                if nnz > k and 2*k > nnz:
                    break
                elif nnz < k/2:
                    r = tmp_ratio
                else:
                    l = tmp_ratio
            indexes = indexes 
            values = tensor.data[indexes] 
            RedSyncCompressor.residuals[name].data = tensor.data + 0.0 
            RedSyncCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = RedSyncCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = RedSyncCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[RedSyncCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 