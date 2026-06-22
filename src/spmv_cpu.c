#include "../include/spmv_cpu.h"

#include <stdlib.h>

void spmv_cpu_coo(const int *__restrict__ I,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int nz,
                  const float *__restrict__ X,
                  float *__restrict__ Y)
{
    int i;

    for (i = 0; i < nz; ++i) Y[I[i]] += val[i] * X[J[i]];
}

void spmv_cpu_csr(const int *__restrict__ O,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int M,
                  const float *__restrict__ X,
                  float *__restrict__ Y)
{
    int i, j;

    for (i = 0; i < M; ++i)
    {
        int start = O[i];
        int end = O[i + 1];

        for (j = start; j < end; ++j) Y[i] += val[j] * X[J[j]];
    }
}
