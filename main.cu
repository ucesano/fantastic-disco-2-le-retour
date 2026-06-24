#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <mpi.h>
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

#define CHECK_CUDA(call)                                  \
    do {                                                  \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess) {                         \
            std::cerr << "CUDA error in " << __FILE__     \
                      << ":" << __LINE__ << " : "         \
                      << cudaGetErrorString(err) << "\n"; \
            MPI_Abort(MPI_COMM_WORLD, err);               \
        }                                                 \
    } while (0)

#define CHECK_MPI(call)                                   \
    do {                                                  \
        int err = (call);                                 \
        if (err != MPI_SUCCESS) {                         \
            char err_str[MPI_MAX_ERROR_STRING];           \
            int err_len;                                  \
            MPI_Error_string(err, err_str, &err_len);     \
            std::cerr << "MPI error in "                  \
                      << __FILE__ << ":" << __LINE__      \
                      << " : " << err_str << "\n";        \
            MPI_Abort(MPI_COMM_WORLD, err);               \
        }                                                 \
    } while (0)

#define OWNER(row, P) \
    ((row) % (P))

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

inline void display_card_informations(const int devCount, std::ostream& os = std::cout)
{
    std::ostringstream oss;

    oss << "There are " << devCount << " CUDA devices.\n";

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
    CHECK_MPI(MPI_Init(&argc, &argv));

    int rank, P;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &P));

    // Abort if only 1 task.
    if (P < 2) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    // Mersenne Twister Initialization.
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    init_by_array(init, length);

    // Checking for CUDA capable devices.
    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));

    if (dev_count == 0)
    {
        fprintf(stderr, "No CUDA device found.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Affecting CUDA capable device to rank.
    int local_dev = rank % dev_count;
    CHECK_CUDA(cudaSetDevice(local_dev));

    // Executable must have a MTX file as arg.
    if (argc < 2)
    {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " matrix.mtx\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Opening MTX File.
    FILE *f = NULL;
    if ((f = fopen(argv[1], "r")) == NULL)
    {
        std::cerr << "Could not open MTX file.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int g_M;
    int g_N;
    int g_nz;

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        std::cerr << "Could not process Matrix Market banner.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (mm_read_mtx_crd_size(f, &g_M, &g_N, &g_nz) != 0)
    {
        std::cerr << "Unsupported Matrix.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::vector<int>   I;
    std::vector<int>   J;
    std::vector<float> val;

    int x, y;
    float z;
    for (int i = 0; i < g_nz; ++i)
    {
        fscanf(f, "%d %d %g\n", &x, &y, &z);
        --x;
        --y;

        if (OWNER(x, P) == rank)
        {
            I.push_back(x);
            J.push_back(y);
            val.push_back(z);
        }

        if (mm_is_symmetric(matcode) && x != y)
        {
            if (OWNER(y, P) == rank)
            {
                I.push_back(y);
                J.push_back(x);
                val.push_back(z);
            }
        }
    }

    fclose(f);

    int l_nz = I.size();

    CHECK_MPI(MPI_Finalize());

    return 0;
}
