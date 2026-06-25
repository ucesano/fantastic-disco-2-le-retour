#include "../include/spmv_cpu.h"

#include <stdio.h>
#include <stdlib.h>


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

        for (j = start; j < end; ++j)
        {
            Y[i] += val[j] * X[J[j]];
        }
    }
}
