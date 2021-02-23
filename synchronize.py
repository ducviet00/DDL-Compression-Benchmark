from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import Average, Adasum, Sum
from horovod.torch.mpi_ops import init, broadcast
import time
import torch
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

def _allreduce_grad_async(p):
    tensor = p.data.view(-1)
    tensor_compressed, ctx = tensor, None
    stime = time.time()
    handle = allreduce_async_(tensor_compressed, average=True)
    return handle, ctx, stime

def _custom_allreduce_async(p, density, compressor, iter, timer=None):
    tensor = p.data.view(-1)
    with timer(compressor.name + " compressing time", epoch=iter):
        tensor_compressed, ctx, selected_values = compressor.compress(tensor, ratio=density, iteration=iter)
    stime = time.time()
    handle = allreduce_async_(selected_values, average=True)
    return handle, ctx, stime

def _sparse_allreduce_async(p, density, compressor, iter, timer=None):
    tensor = p.data.view(-1)
    with timer(compressor.name + " compressing time", epoch=iter):
        tensor_compressed, ctx, selected_values = compressor.compress(tensor, ratio=density)
    indexes = ctx
    stime = time.time()
    handle = allgather_async(selected_values)
    handle_idx = allgather_async(indexes.int())
    return (handle, handle_idx), ctx, stime

def torch_intersect(t1, t2):
    t1= set(t1.unique())
    t2= set(t2.unique())    
    return t1.intersection(t2)

def gtopk_sparse_allreduce(comm, values, indexes, density, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = size()
    rank = rank()

    tensor = values
    k = indexes.size[0]
    original_indexes = indexes

    send_values = torch.cat((indexes, values))
    recv_values = np.zeros_like(send_values)

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                tmp_indexes = recv_values[0:k]
                tmp_values = recv_values[k:2*k]

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = torch.topk(torch.abs(tensor.data), k=k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
    else:
        send_values = recv_values[0:2*k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes, indexes, assume_unique=False, return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes # final selected values and indexes


def post_synchronize(tensor, handle, ctx, density, method="none", timer=None):
    num_of_workers = size()
    """
    Structured and random block commnunicate implemnetion
    """
    if method == 'structured' or method == 'randomblock':
        assert type(handle) is not tuple
        assert type(ctx) is tuple
        output = synchronize(handle)
        etime = time.time()
        with timer(method + " decompressing time"):
            start_idx, end_idx = ctx[0], ctx[1]
            new_grad = tensor.data.view(-1)
            new_grad.fill_(0.0)
            new_grad[start_idx:end_idx] = output

        return new_grad, etime
    """
    Top-K and random-K communicate implemention
    """
    if method == "topk" or method == 'randomk' \
        or method == 'redsync' or method == 'gaussian':
        assert type(handle) is tuple
        handle, handle_idx = handle[0], handle[1]

        output = synchronize(handle)
        all_indexes = synchronize(handle_idx)
        etime = time.time()
        with timer(method + " decompressing time"):
            new_grad = tensor.data.view(-1)
            new_grad.fill_(0.0)
            numel = output.size(0)
            real_num_values = numel//num_of_workers

            for i in range(num_of_workers):
                values = output.data[i*real_num_values:(i+1)*real_num_values] 
                indexes = all_indexes.data[i*real_num_values:(i+1)*real_num_values].long()
                new_grad[indexes] += values

            new_grad /= num_of_workers
            return new_grad, etime

    """
    Non compression - pure all-reduce communicate
    """
    if method == "none":
        new_grad = synchronize(handle)
        etime = time.time()
        return new_grad, etime
