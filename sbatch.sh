#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem=1G

#SBATCH --job-name=mpi_spmv_test
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

export CUDAHOSTCXX=$(which gcc)

module load OpenMPI
module load CUDA/11.8.0

gcc --version
mpicc --version

module list | grep -i cuda

make

mpirun -np 4 ./bin/spmv mtx/F1.mtx
