#ifndef SPMV_GPU_CUH
#define SPMV_GPU_CUH

__global__
void spmv_gpu_csr_opt(const int *__restrict__ O,
                         const int *__restrict__ J,
                         const float *__restrict__ val,
                         const int M,
                         const float *__restrict__ X,
                         float *__restrict__ Y);

#endif
