#ifndef MM_FMT_H
#define MM_FMT_H

void mm_sort_coo(int *__restrict__ I,
                 int *__restrict__ J,
                 float *__restrict__ val,
                 const int nz);

void mm_coo_to_csr_row_ptr(const int *__restrict__ I,
                           const int nz,
                           const int M,
                           int *__restrict__ O);

#endif
