#include "../include/bench.cuh"

#include "../include/spmv_gpu.cuh"

#include <sstream>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif

#include "../include/spmv_cpu.h"

#ifdef __cplusplus
}
#endif

void print_results(const struct results& res, const char * file, const char * fmt, std::ostream& os)
{
    std::ostringstream oss;

    oss << "SpMV Results:\n\n";
    oss << "   File . . . . . . . . . . . . . : " << file << "\n";
    oss << "   format . . . . . . . . . . . . : " << fmt << '\n';
    oss << "   time . . . . . . . . . . . . . : " << res.exec_time << '\n';
    oss << "   gflops . . . . . . . . . . . . : " << res.gflops << '\n';
    oss << "   correct  . . . . . . . . . . . : " << (res.is_correct ? "yes" : "no") << '\n';

    os << oss.str() << std::endl;
}

struct results spmv_gpu_csr_opt_prof(const int *__restrict__ O,
                                     const int *__restrict__ J,
                                     const float *__restrict__ val,
                                     const int M,
                                     const int N,
                                     const int nz,
                                     const int *__restrict__ row_nz,
                                     const float *__restrict__ X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dO, *dJ;
    float *dval, *dX, *dY;

    cudaMalloc(&dO, (M + 1) * sizeof(int));
    cudaMalloc(&dJ, nz * sizeof(int));
    cudaMalloc(&dval, nz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dO, O, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (M * 32 + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMem = threadsPerBlock * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_csr_opt<<<numBlocks, threadsPerBlock, sharedMem>>>(dO, dJ, dval, M, dX, dY);
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_csr_opt<<<numBlocks, threadsPerBlock, sharedMem>>>(dO, dJ, dval, M, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz - M) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dO);
    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_csr(O, J, val, M, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-5f;
    float tol_rel = 1e-5f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}
