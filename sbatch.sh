#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

#SBATCH --job-name=mpi_spmv_test
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

module load OpenMPI
module load CUDA/11.8.0

hostname
# nvidia-smic
# ompi_info --parsable --all | grep mpi_built_with_cuda_support
# ompi_info | grep -iE 'ucx|cuda'

make

mpirun -np 4 ./bin/spmv mtx/boyd2.mtx
