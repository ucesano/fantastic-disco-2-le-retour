#ifndef BENCH_H
#define BENCH_H

#define WARMUP 10
#define ITERATION 30

struct results
{
    float exec_time;
    //float mem_bandwidth;
    float gflops;
    char   is_correct;
};

void print_results(const struct results res, const char * fmt);

struct results spmv_gpu_coo_prof(const int *__restrict__ I,
                                 const int *__restrict__ J,
                                 const float *__restrict__ val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *__restrict__ row_nz,
                                 const float *__restrict__ X);

struct results spmv_gpu_csr_prof(const int *__restrict__ O,
                                 const int *__restrict__ J,
                                 const float *__restrict__ val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *__restrict__ row_nz,
                                 const float *__restrict__ X);

struct results spmv_gpu_csr_opt_prof(const int *__restrict__ O,
                                     const int *__restrict__ J,
                                     const float *__restrict__ val,
                                     const int M,
                                     const int N,
                                     const int nz,
                                     const int *__restrict__ row_nz,
                                     const float *__restrict__ X);

#endif
