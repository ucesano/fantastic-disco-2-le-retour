#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

#SBATCH --job-name=mpi_spmv_test
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

module load CUDA/11.8.0
module load OpenMPI

hostname
# nvidia-smi
# ompi_info --parsable --all | grep mpi_built_with_cuda_support
# ompi_info | grep -iE 'ucx|cuda'

make

mpirun -np 2 ./bin/spmv mtx/sell.mtx
