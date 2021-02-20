from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import Average, Adasum, Sum
from horovod.torch.mpi_ops import init, broadcast

import torch
import numpy as np

def _allreduce_grad_async(p):
    tensor = p.data.view(-1)
    tensor_compressed, ctx = tensor, None
    handle = allreduce_async_(tensor_compressed, average=True)
    return handle, ctx

def _custom_allreduce_async(p, density, compressor, iter, timer=None):
    tensor = p.data.view(-1)
    with timer(compressor.name + " compressing", epoch=iter):
        tensor_compressed, ctx, selected_values = compressor.compress(tensor, ratio=density, iteration=iter)
    handle = allreduce_async_(selected_values, average=True)
    return handle, ctx

def _sparse_allreduce_async(p, density, compressor, iter, timer=None):
    tensor = p.data.view(-1)
    with timer(compressor.name + " compressing", epoch=iter):
        tensor_compressed, ctx, selected_values = compressor.compress(tensor, ratio=density)
    indexes = ctx
    handle = allgather_async(selected_values)
    handle_idx = allgather_async(indexes.int())
    return (handle, handle_idx), ctx 


def post_synchronize(tensor, handle, ctx, density, method="none", timer=None):
    num_of_workers = size()
    """
    Structured and random block commnunicate implemnetion
    """
    if method == 'structured' or method == 'randomblock':
        assert type(handle) is not tuple
        assert type(ctx) is tuple
        start_idx, end_idx = ctx[0], ctx[1]
        new_grad = tensor.data.view(-1)
        output = synchronize(handle)
        new_grad.fill_(0.0)
        new_grad[start_idx:end_idx] = output

        return new_grad
    """
    Top-K and random-K communicate implemention
    """
    if method == "topk" or method == 'randomk':
        assert type(handle) is tuple
        handle, handle_idx = handle[0], handle[1]

        output = synchronize(handle)
        all_indexes = synchronize(handle_idx)
        new_grad = tensor.data.view(-1)
        new_grad.fill_(0.0)
        numel = output.size(0)
        real_num_values = numel//num_of_workers

        for i in range(num_of_workers):
            values = output.data[i*real_num_values:(i+1)*real_num_values] 
            indexes = all_indexes.data[i*real_num_values:(i+1)*real_num_values].long()
            new_grad[indexes] += values

        new_grad /= num_of_workers
        return new_grad

    """
    Non compression - pure all-reduce communicate
    """
    if method == "none":
        new_grad = synchronize(handle)
        return new_grad
