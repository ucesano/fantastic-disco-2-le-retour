#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "include/mmio.h"
#include "include/mmfmt.h"

#include "include/mt19937ar.h"

#ifdef __cplusplus
}
#endif

#include "include/bench.cuh"

static void print_device_properties(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("  Memory Clock rate:           %.0f Mhz\n", devProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:            %d bit\n",devProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:       %7.3f GB/s\n",2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);

    printf("  Multiprocessors:             %3d\n",devProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor:  %d\n",devProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",devProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           devProp.maxGridSize[0], devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("  Total amount of shared memory per block:       %zu bytes\n", devProp.sharedMemPerBlock);
    putchar(10);
}

static void display_card_informations(void)
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        print_device_properties(devProp);
    }
}

int main(int argc, char ** argv)
{
    int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    int ret_code;

    MM_typecode matcode;
    FILE *f;
    int i;
    int M, N, nz;
    int *I, *O, *J, *row_nz;
    float *val, *X;

    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    init_by_array(init, length);

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(EXIT_FAILURE);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* finding out size of sparse matrix. */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(EXIT_FAILURE);

    /* reserving memory for COO representation. */
    I = (int *) malloc(nz * (1 + mm_is_symmetric(matcode)) * sizeof(int));
    J = (int *) malloc(nz * (1 + mm_is_symmetric(matcode)) * sizeof(int));
    val = (float *) malloc(nz * (1 + mm_is_symmetric(matcode)) * sizeof(float));

    if (mm_is_general(matcode))
    {
        for(i = 0; i < nz; ++i)
        {
            fscanf(f, "%d %d %g\n", &I[i], &J[i], &val[i]);
            I[i]--;
            J[i]--;
        }
    }
    else if (mm_is_symmetric(matcode))
    {
        int extra = 0;

        for(i = 0; i < nz; ++i)
        {
            fscanf(f, "%d %d %g\n", &I[i], &J[i], &val[i]);
            I[i]--;
            J[i]--;

            if (I[i] != J[i])
            {
                I[nz + extra] = J[i];
                J[nz + extra] = I[i];
                val[nz + extra] = val[i];

                extra++;
            }
        }

        nz += extra;

        int *tI, *tJ;
        float *tval;

        tI = (int *)realloc(I, nz * sizeof(int));
        tJ = (int *)realloc(J, nz * sizeof(int));
        tval = (float *)realloc(val, nz * sizeof(float));

        if (tI != NULL && tJ != NULL && tval != NULL)
        {
            I = tI;
            J = tJ;
            val = tval;
        }
    }

    fclose(f);

    mm_sort_coo(I, J, val, nz);

    O = (int *)calloc((M + 1), sizeof(int));
    mm_coo_to_csr_row_ptr(I, nz, M, O);

    row_nz = (int *)calloc(M, sizeof(int));
    for (i = 0; i < nz; ++i) row_nz[I[i]]++;

    X = (float *) malloc(N * sizeof(float));
    for (i = 0; i < N; i++) X[i] = (float)genrand_real1() + 1.f;

    struct results coo     = spmv_gpu_coo_prof(I, J, val, M, N, nz, row_nz, X);
    struct results csr     = spmv_gpu_csr_prof(O, J, val, M, N, nz, row_nz, X);
    struct results csr_opt = spmv_gpu_csr_opt_prof(O, J, val, M, N, nz, row_nz, X);

    free(row_nz);
    free(X);
    free(O);
    free(I);
    free(J);
    free(val);

    display_card_informations();

    fprintf(stdout, "file: %s\n", argv[1]);
    print_results(coo, "coo");
    print_results(csr, "csr");
    print_results(csr_opt, "csr (warp)");

    return ret_code;
}
