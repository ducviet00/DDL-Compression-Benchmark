#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=05:00:00
#$ -N res50_64
#$ -o ./logs/$JOB_ID.$JOB_NAME.log
#$ -j y
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299

DENSITY=0.01
METHOD="randomk"
TENSOR_SIZE=2500000
NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

LOG_DIR="./logs/G${NUM_PROCS}"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
cat $SGE_JOB_HOSTLIST > ${LOG_DIR}/$JOB_ID.$JOB_NAME.nodes.list


MPIOPTS="-np ${NUM_PROCS} --hostfile $SGE_JOB_HOSTLIST --oversubscribe -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0" #-x NCCL_DEBUG=INFO"
mpirun ${MPIOPTS} python3 benchmark_comm.py --method $METHOD --density $DENSITY --size $TENSOR_SIZE
