import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank
import utils

class GaussianCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'gaussian'
    @staticmethod
    def clear():
        GaussianCompressor.residuals = {}
        GaussianCompressor.sparsities = []
        GaussianCompressor.zero_conditions = {}
        GaussianCompressor.values = {} 
        GaussianCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=3, ratio=0.05):
        gpu_rank = rank()
        start = time.time()
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = torch.nonzero(one_indexes).data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            #one_indexes = abs_tensor > right_thres
            #indexes = one_indexes.nonzero().data.squeeze().view(-1)
            #indexes = indexes #[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0 
            GaussianCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = GaussianCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = GaussianCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[GaussianCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 