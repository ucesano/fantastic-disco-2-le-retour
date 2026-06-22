#include "../include/mmfmt.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct coo_entry
{
    int row;
    int col;
    float val;
};

static int coo_cmp(const void *a, const void *b)
{
    const struct coo_entry *ea = (const struct coo_entry *)a;
    const struct coo_entry *eb = (const struct coo_entry *)b;

    return (ea->row != eb->row) ? ea->row - eb->row : ea->col - eb->col;
}

void mm_sort_coo(int *__restrict__ I,
                 int *__restrict__ J,
                 float *__restrict__ val,
                 const int nz)
{
    int i;

    struct coo_entry *coo = (struct coo_entry *) malloc(nz * sizeof(struct coo_entry));

    for (i = 0; i < nz; ++i)
    {
        coo[i].row = I[i];
        coo[i].col = J[i];
        coo[i].val = val[i];
    }

    qsort(coo, nz, sizeof(struct coo_entry), coo_cmp);

    for (i = 0; i < nz; ++i)
    {
        I[i]   = coo[i].row;
        J[i]   = coo[i].col;
        val[i] = coo[i].val;
    }

    free(coo);
}

void mm_coo_to_csr_row_ptr(const int *__restrict__ I,
                           const int nz,
                           const int M,
                           int *__restrict__ O)
{
    int i;

    for (i = 0; i < nz; ++i) O[I[i] + 1]++;
    for (i = 1; i <= M; ++i) O[i] += O[i - 1];
}
