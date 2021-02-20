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
init()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Communicate across gpus")
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--nloop', type=int, default=100)
    parser.add_argument('--size', type=int, default='2500000')
    parser.add_argument('--density', type=float, default='0.01')
    args = parser.parse_args()

    for iter in range(args.nloop):

        grad_tensor = torch.rand(args.size)
        print(f"[rank: {rank()}][iter: {iter}] Grad matrix before: {torch.sum(grad_tensor)}")

        if args.method == 'structured' or args.method == 'randomblock':
            handle, ctx = _custom_allreduce_async(grad_tensor, args.density, compressors[args.method], iter=iter)
            new_tensor  = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method)

        elif args.method == 'topk' or args.method == 'randomk':
            handle, ctx = _sparse_allreduce_async(grad_tensor, args.density, compressors[args.method])
            new_tensor  = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method)

        elif args.method == 'none' or args.method is None:
            handle, ctx = _allreduce_grad_async(grad_tensor)
            new_tensor = post_synchronize(grad_tensor, handle, ctx, args.density, method=args.method)
        
        print(f"[rank: {rank()}][iter: {iter}] Grad matrix after: {torch.sum(grad_tensor)}")
        # print(f"[rank: {rank()}][iter: {iter}] New tensor after: {torch.sum(new_tensor)}")