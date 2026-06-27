#pragma region includes

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

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

#pragma endregion includes

#pragma region helpers

#pragma region misc_helpers

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

#pragma endregion misc_helpers

#pragma region partitioning_helpers

static inline int owner(const int i, const int P) { return i % P; }
static inline int global_to_local(const int i, const int P) { return i / P; }
static inline int local_to_global(const int l, const int p, const int P) { return l * P + p; }

#pragma endregion partitioning_helpers

#pragma region info_helpers

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

#pragma endregion info_helpers

#pragma endregion helpers

#pragma region debug

void print_coo_local(const std::vector<int>& I,
                     const std::vector<int>& J,
                     const std::vector<float>& val,
                     int P, int rank = -1) {
    const std::size_t nnz = I.size();
    if (J.size() != nnz || val.size() != nnz) {
        std::cerr << "[print_coo_local] SIZE MISMATCH\n";
        return;
    }

    std::cout << "rank " << rank << ": " << nnz << " local nonzeros (P=" << P << ")\n";

    std::cout << "I     = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << I[k];
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "owner = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << I[k] % P;          // owner (should all == rank)
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "local = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << I[k] / P;          // local row
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "J     = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << J[k];
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "val   = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << val[k];
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";
}

#pragma endregion debug

int main(int argc, char ** argv)
{
    #pragma region init

    CHECK_MPI(MPI_Init(&argc, &argv));

    int p, P;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &p));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &P));

    // Abort if only 1 task.
    if (P < 2) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    // Mersenne Twister Initialization.
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    init_by_array(init, length);

    // Checking for CUDA capable devices.
    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));

    // std::cout << "p=" << p << "\n";
    // display_card_informations(dev_count);

    if (dev_count == 0)
    {
        fprintf(stderr, "No CUDA device found.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Affecting CUDA capable device to p.
    int local_dev = p % dev_count;
    CHECK_CUDA(cudaSetDevice(local_dev));

    // Executable must have a MTX file as arg.
    if (argc < 2)
    {
        if (p == 0) std::cerr << "Usage: " << argv[0] << " matrix.mtx\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    #pragma endregion init

    int g_M  = 0;
    int g_N  = 0;
    int g_nz = 0;

    #pragma region loading_file

    std::vector<int> I;
    std::vector<int> J;
    std::vector<float> val;

    if (p == 0)
    {
        FILE *f = NULL;
        if ((f = fopen(argv[1], "r")) == NULL)
        {
            std::cerr << "Could not open MTX file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MM_typecode matcode;

        if (mm_read_banner(f, &matcode) != 0)
        {
            std::cerr << "Could not process Matrix Market banner." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (mm_read_mtx_crd_size(f, &g_M, &g_N, &g_nz) != 0)
        {
            std::cerr << "Unsupported Matrix." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < g_nz; ++i)
        {
            int x;
            int y;
            float z;

            fscanf(f, "%d %d %g\n", &x, &y, &z);

            // Going from 1-based to 0-based.
            --x;
            --y;

            I.push_back(x);
            J.push_back(y);
            val.push_back(z);

            if (mm_is_symmetric(matcode))
            {
                I.push_back(y);
                J.push_back(x);
                val.push_back(z);
            }
        }

        if (!(I.size() == J.size() && J.size() == val.size()))
        {
            std::cerr << "Ill-formed matrix." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        g_nz = I.size();

        mm_sort_coo(I.data(), J.data(), val.data(), g_nz);
    }

    CHECK_MPI(MPI_Bcast(&g_M, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&g_N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&g_nz, 1, MPI_INT, 0, MPI_COMM_WORLD));

    #pragma endregion loading_file

    if (p == 0) print_coo_local(I, J, val, P, p);

    #pragma region cleaning_up

    CHECK_MPI(MPI_Finalize());

    #pragma endregion cleaning_up

    return EXIT_SUCCESS;
}
