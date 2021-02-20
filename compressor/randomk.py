import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank

class RandomKCompressor():
    """
    """
    residuals = None
    name = 'randomk'
    
    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        with torch.no_grad():
            if RandomKCompressor.residuals is None:
                RandomKCompressor.residuals = torch.zeros_like(tensor.data)
                
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes]

            tensor.data.add_(RandomKCompressor.residuals.data)
            RandomKCompressor.residuals.data = tensor.data + 0.0
            RandomKCompressor.residuals.data[indexes] = 0.

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 