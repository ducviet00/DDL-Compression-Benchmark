import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank

class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = None
    name = 'topk'
    
    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        with torch.no_grad():
            if TopKCompressor.residuals is None:
                TopKCompressor.residuals = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
                
            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            tensor.data.add_(TopKCompressor.residuals.data)
            TopKCompressor.residuals.data = tensor.data + 0.0
            TopKCompressor.residuals.data[indexes] = 0.

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 