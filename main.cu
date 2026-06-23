#include <iostream>
#include <sstream>

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

inline void print_device_properties(const cudaDeviceProp& dev, std::ostream& os)
{
    const double MHz       = 1e-3;
    const double GBpsScale = 1.0e6;

    const int busBytes             = dev.memoryBusWidth / 8;
    const double peakBandwidthGBps = 2.0 * dev.memoryClockRate * busBytes / GBpsScale;

    os << "\n"
       << "   Name . . . . . . . . . . . . . : " << dev.name << '\n'
       << "   Major revision number  . . . . : " << dev.major << '\n'
       << "   Minor revision number  . . . . : " << dev.minor << '\n'
       << "   Memory Clock rate (MHz)  . . . : "
       << static_cast<long>(dev.memoryClockRate * MHz) << '\n'
       << "   Memory Bus Width (bits)  . . . : " << dev.memoryBusWidth << '\n'
       << "   Peak Memory Bandwidth (GB/s) . : ";
    os.setf(std::ios_base::fixed, std::ios_base::floatfield);
    os.precision(3);
    os << peakBandwidthGBps << '\n';
    os.unsetf(std::ios_base::floatfield);
    os << "   Multiprocessors  . . . . . . . : " << dev.multiProcessorCount << '\n'
       << "   Max threads per block  . . . . : " << dev.maxThreadsPerBlock << '\n'
       << "   Shared memory per block (bytes): " << dev.sharedMemPerBlock << '\n';
}

inline void display_card_informations(std::ostream& os = std::cout)
{
    std::ostringstream oss;

    int devCount;
    cudaGetDeviceCount(&devCount);

    oss << "CUDA Device Query...\nThere are " << devCount << " CUDA devices.\n";

    for (int i = 0; i < devCount; ++i)
    {
        oss << "\nCUDA Device #" << i << '\n';
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        print_device_properties(devProp, oss);
    }

    os << oss.str() << std::endl;
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
