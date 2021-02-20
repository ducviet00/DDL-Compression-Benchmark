import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank


class StructuredCompressor():
    """
    A novel method
    """
    residuals = None
    name = 'structured'
    
    @staticmethod
    def compress(tensor, name=None, ratio=0.05, iteration=None):
        assert iteration is not None, print("NO ITERATION ?")
        gpu_rank = rank()
        start = time.time()
        with torch.no_grad():

            if StructuredCompressor.residuals is None:
                StructuredCompressor.residuals = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            idx = iteration % (numel // k + 1)
            
            tensor.data.add_(StructuredCompressor.residuals.data)
            values = tensor[idx*k:(idx+1)*k].clone()

            StructuredCompressor.residuals.data = tensor.data + 0.0
            StructuredCompressor.residuals.data[idx*k:(idx+1)*k] = 0. 
            return tensor, (idx*k, (idx+1)*k), values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 