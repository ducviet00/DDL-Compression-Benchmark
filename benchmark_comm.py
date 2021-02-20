import os
import argparse
import time
import psutil
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast
import torch
import numpy
from synchronize import post_synchronize
from synchronize import _allreduce_grad_async, _sparse_allreduce_async, _custom_allreduce_async
from compressor import compressors
from timer import Timer
init()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Benchmark Communicate across gpus")
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--nloop', type=int, default=100)
    parser.add_argument('--size', type=int, default='2500000')
    parser.add_argument('--density', type=float, default='0.01')
    parser.add_argument('--nwpernode', type=int, default=4)
    args = parser.parse_args()
    
    timer = Timer()
    rank = rank()
    torch.cuda.set_device(rank%args.nwpernode)

    for iter in range(args.nloop):
        with timer("one iteration time", epoch=iter):
            grad_tensor = torch.rand(args.size)
            # print(f"[rank: {rank()}][iter: {iter}] Grad tensor before: {torch.sum(grad_tensor)}")
            with timer("compress and sync", epoch=iter):
                if args.method == 'structured' or args.method == 'randomblock':
                    handle, ctx = _custom_allreduce_async(grad_tensor, args.density, compressors[args.method], iter=iter, timer=timer)
                    new_tensor  = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method, timer=timer)

                elif args.method == 'topk' or args.method == 'randomk':
                    handle, ctx = _sparse_allreduce_async(grad_tensor, args.density, compressors[args.method], iter=iter, timer=timer)
                    new_tensor  = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method, timer=timer)

                elif args.method == 'none' or args.method is None:
                    handle, ctx = _allreduce_grad_async(grad_tensor, timer=timer)
                    new_tensor = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method, timer=timer)
        
            # print(f"[rank: {rank()}][iter: {iter}] Grad tensor after: {torch.sum(grad_tensor)}")
            # print(f"[rank: {rank()}][iter: {iter}] New tensor after: {torch.sum(new_tensor)}")

    if rank == 0:
        print("\n" + timer.summary() + "\n")
        with open(f"logs/SUMMARY_{args.method}.txt", "w") as f:
            f.write(f"[Rank {rank}]:\n")
            f.write(timer.summary())
            f.write("\n")