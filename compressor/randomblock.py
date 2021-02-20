import torch
import numpy as np
import time
import math
import random

from horovod.torch.mpi_ops import rank


class RandomBlockCompressor():
    """
    A novel method
    """
    residuals = None
    random.seed(42)
    name = 'randomblock'
    
    @staticmethod
    def compress(tensor, name=None, ratio=0.05, iteration=None):
        assert iteration is not None, print("NO ITERATION ?")
        with torch.no_grad():

            if RandomBlockCompressor.residuals is None:
                RandomBlockCompressor.residuals = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            idx = random.randint(0, numel-k)
            
            tensor.data.add_(RandomBlockCompressor.residuals.data)
            values = tensor[idx:idx+k].clone()

            RandomBlockCompressor.residuals.data = tensor.data + 0.0
            RandomBlockCompressor.residuals.data[idx:idx+k] = 0. 
            return tensor, (idx, idx+k), values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 