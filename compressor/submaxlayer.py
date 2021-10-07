
import torch
import numpy as np
import time
import math
from horovod.torch.mpi_ops import rank, size
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import synchronize
import utils

class SubMaxbyLayerCompressor():
    """
    A novel method
    """
    residuals = {}
    stop_indices = []
    start_indices = []
    layer_sizes = []
    layer_subsize = []
    name = 'submax_by_layer'
    # logger.info("opening txt file to write compression time")
    ftime = open("logs/LAYERSUBMAX_compression_time.txt", "a",  buffering=1)
    
    @staticmethod
    def compress(tensor, name=None, ratio=0.01):
        """
        max_indices: Tensor size: number of nodes
        """
        # with tracking("COMPRESSION TIME"):
        # assert max_indices is not None, print("WHERE MAX ??")
        gpu_rank = rank()
        start = time.time()
        with torch.no_grad():
            # print(f"{gpu_rank} tensor before compression:", tensor.size())
            # # Residual technique be commented SubMaxbyLayerCompressor.residuals
            # --------------------------------------------------------------------------
            if name not in SubMaxbyLayerCompressor.residuals:
                SubMaxbyLayerCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # assert sum(SubMaxbyLayerCompressor.layer_sizes) == tensor.numel()
            # local_slice = [slice(s, s+k) for s, k in zip(
                # SubMaxbyLayerCompressor.start_indices, SubMaxbyLayerCompressor.layer_subsize)]
            # local_indexer = np.r_[tuple(local_slice)]
            # SubMaxbyLayerCompressor.residuals[name].data = tensor.data + 0.0
            # SubMaxbyLayerCompressor.residuals[name].data[local_indexer] = 0. 
            # assert torch.sum(tensor.data[local_indexer]) != 0
            # assert torch.sum(SubMaxbyLayerCompressor.residuals[name].data[local_indexer]) == 0
            # --------------------------------------------------------------------------
            handle_start = allgather_async(torch.Tensor([SubMaxbyLayerCompressor.start_indices]))
            # handle_stop = allgather_async(torch.Tensor([SubMaxbyLayerCompressor.stop_indices]))
            layer_subsize = SubMaxbyLayerCompressor.layer_subsize*size()
            start_indices = synchronize(handle_start)
            start_indices = start_indices.data.view(-1)
            # stop_indices = synchronize(handle_stop)
            # stop_indices = stop_indices.data.view(-1)
            intervals = [[s.item(), s.item()+k] for s, k in zip(start_indices, layer_subsize)]
            slices = SubMaxbyLayerCompressor.merge(intervals)
            # slices = [slice(s.item(), s.item()+k) for s, k in zip(start_indices, layer_subsize)]

            indexer = np.r_[tuple(slices)]
            indexer = np.unique(indexer)
            # print("indexer", indexer.shape)
            values = tensor[indexer]
            SubMaxbyLayerCompressor.layer_sizes = []
            SubMaxbyLayerCompressor.layer_subsize = []
            SubMaxbyLayerCompressor.start_indices = []
            # SubMaxbyLayerCompressor.stop_indices = []
            # print(f"{gpu_rank} tensor after compression:", type(values), torch.sum(values.abs()), values.size())
            # print(f"[rank {gpu_rank}]", start_indicies.shape)
            SubMaxbyLayerCompressor.ftime.write(f"[rank {gpu_rank}] \tCOMPRESSION TIME: {-start + time.time()} \n")
            return tensor, indexer, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z

    @staticmethod   
    def find_name(name):
        for key in SubMaxbyLayerCompressor.residuals.keys():
            if name in key:
                return key
    @staticmethod
    def select_sub_max(tensor, name, ratio, iter):
        with torch.no_grad():
            numel = tensor.numel()
            if iter > 0:
                name = SubMaxbyLayerCompressor.find_name(name)
                s = sum(SubMaxbyLayerCompressor.layer_sizes)
                tensor_serialized = torch.add(SubMaxbyLayerCompressor.residuals[name].data[s:s+numel], tensor.view(-1)).abs()
            else:
                tensor_serialized = tensor.view(-1).abs()

            k = max(int(numel * ratio), 1)
            a_cumsum = torch.cumsum(tensor_serialized, dim=0)
            k_cumsum = a_cumsum[k:] - a_cumsum[:-k]
            start_idx = sum(SubMaxbyLayerCompressor.layer_sizes) + torch.argmax(k_cumsum).item()
            SubMaxbyLayerCompressor.layer_subsize.append(k)
            SubMaxbyLayerCompressor.start_indices.append(start_idx)
            # SubMaxbyLayerCompressor.stop_indices.append(start_idx+k)
            SubMaxbyLayerCompressor.layer_sizes.append(numel)
            if iter > 0:
                SubMaxbyLayerCompressor.residuals[name].data[s:s+numel] = tensor.view(-1).data + 0.
                SubMaxbyLayerCompressor.residuals[name].data[start_idx:start_idx+k] = 0.0
    
    @staticmethod
    def merge(intervals):
        # merge overlapping block
        # O(nlogn) solution
        intervals.sort(key=lambda x: x[0])
        slices = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not slices or slices[-1].stop < interval[0]:
                slices.append(slice(interval[0], interval[1]))
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                slices[-1] = slice(slices[-1].start, max(slices[-1].stop, interval[1]))

        return slices
