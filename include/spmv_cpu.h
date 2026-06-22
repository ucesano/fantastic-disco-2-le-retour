#ifndef SPMV_CPU_H
#define SPMV_CPU_H

void spmv_cpu_coo(const int *__restrict__ I,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int nz,
                  const float *__restrict__ X,
                  float *__restrict__ Y);

void spmv_cpu_csr(const int *__restrict__ O,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int M,
                  const float *__restrict__ X,
                  float *__restrict__ Y);

#endif
