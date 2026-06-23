#include "../include/spmv_gpu.cuh"

#include <stdlib.h>

__global__
void spmv_gpu_csr_opt(const int   *__restrict__ O,
                         const int   *__restrict__ J,
                         const float *__restrict__ val,
                         const int    M,
                         const float *__restrict__ X,
                         float       *__restrict__ Y)
{
    extern __shared__ float vals[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / 32;
    int lane = tid & (32 - 1);

    int row = wid;

    vals[threadIdx.x] = 0.f;

    if (row < M)
    {
        int row_start = O[row];
        int row_end   = O[row + 1];

        for (int j = row_start + lane; j < row_end; j += 32)
            vals[threadIdx.x] += val[j] * X[J[j]];
    }

    __syncwarp();

    if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16]; __syncwarp();
    if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8]; __syncwarp();
    if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4]; __syncwarp();
    if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2]; __syncwarp();
    if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1]; __syncwarp();

    if (lane == 0)
        Y[row] = vals[threadIdx.x];
}
