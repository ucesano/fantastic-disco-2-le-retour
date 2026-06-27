#pragma region includes

#include <iostream>
#include <iomanip>
#include <numeric>
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
#include <mpi-ext.h>
#include <cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

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

void pack_by_rank(std::vector<int>& I, std::vector<int>& J, std::vector<float>& val, int P)
{
    const std::size_t nnz = I.size();
    if (J.size() != nnz || val.size() != nnz) {
        std::cerr << "[pack_by_rank] SIZE MISMATCH\n";
        return;
    }

    std::vector<std::size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);

    std::stable_sort(perm.begin(), perm.end(),
        [&](std::size_t a, std::size_t b) {
            return owner(I[a], P) < owner(I[b], P);
        });

    std::vector<int>   I2(nnz), J2(nnz);
    std::vector<float> val2(nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
        I2[k]   = I[perm[k]];
        J2[k]   = J[perm[k]];
        val2[k] = val[perm[k]];
    }

    I   = std::move(I2);
    J   = std::move(J2);
    val = std::move(val2);
}

void pack_x_by_rank(std::vector<float>& X, int P)
{
    const int n = (int)X.size();   // = g_N
    std::vector<int> xcount(P, 0), xdispl(P, 0);
    for (int i = 0; i < n; ++i) xcount[i % P]++;
    for (int r = 1; r < P; ++r) xdispl[r] = xdispl[r-1] + xcount[r-1];

    std::vector<float> X2(n);
    std::vector<int> cursor = xdispl;
    for (int i = 0; i < n; ++i)
        X2[cursor[i % P]++] = X[i];   // entry i (column i) -> rank i%P's block

    X = std::move(X2);
}

void scatter_in_place(int g_nz, int g_N, int P, int rank, int*& d_I, int*& d_J, float*& d_val, float*& d_X, int nz, int N)
{
    // 1. counts + displs on rank 0 (needed for the Scatterv send side)
    std::vector<int> ccount(P,0), cdispl(P,0), xcount(P,0), xdispl(P,0);
    if (rank == 0)
    {
        std::vector<int> hI(g_nz);
        cudaMemcpy(hI.data(), d_I, g_nz*sizeof(int), cudaMemcpyDeviceToHost);
        for (int k = 0; k < g_nz; ++k) ccount[hI[k] % P]++;
        for (int r = 1; r < P; ++r) cdispl[r] = cdispl[r-1] + ccount[r-1];
        for (int r = 0; r < P; ++r) xcount[r] = (g_N - r + P - 1) / P;
        for (int r = 1; r < P; ++r) xdispl[r] = xdispl[r-1] + xcount[r-1];
    }

    // 3. temporary recv buffers (sized by the known local counts)
    int   *r_I, *r_J;
    float *r_val, *r_X;
    cudaMalloc(&r_I,   nz * sizeof(int));
    cudaMalloc(&r_J,   nz * sizeof(int));
    cudaMalloc(&r_val, nz * sizeof(float));
    cudaMalloc(&r_X,   N  * sizeof(float));

    // 4. scatter (device ptrs for data, host arrays for counts/displs)
    MPI_Scatterv(d_I,   ccount.data(), cdispl.data(), MPI_INT,   r_I,   nz, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Scatterv(d_J,   ccount.data(), cdispl.data(), MPI_INT,   r_J,   nz, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Scatterv(d_val, ccount.data(), cdispl.data(), MPI_FLOAT, r_val, nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(d_X,   xcount.data(), xdispl.data(), MPI_FLOAT, r_X,   N,  MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 5. free rank 0's full arrays
    if (rank == 0) {
        cudaFree(d_I); cudaFree(d_J); cudaFree(d_val); cudaFree(d_X);
    }

    // 6. rebind caller pointers to the slices
    d_I = r_I;  d_J = r_J;  d_val = r_val;  d_X = r_X;
}


#pragma endregion partitioning_helpers

#pragma region communication_helpers

__global__ void count_per_row(const int* d_I, int nz, int P, int* d_O)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < nz)
    {
        int local_row = d_I[k] / P;
        atomicAdd(&d_O[local_row + 1], 1);
    }
}

#pragma endregion communication_helpers

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
                     const std::vector<float>& X,
                     int P, int rank = -1)
{
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
        std::cout << owner(I[k], P);          // owner (should all == rank)
        if (k + 1 < nnz) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "local = [";
    for (std::size_t k = 0; k < nnz; ++k) {
        std::cout << global_to_local(I[k], P);          // local row
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

    std::cout << "X     = [";
    for (std::size_t k = 0; k < X.size(); ++k) {
        std::cout << X[k];
        if (k + 1 < X.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

void debug_print_device(const int* d_I, const int* d_J,
                        const float* d_val, const float* d_X,
                        int nz, int N, int rank)
{
    std::vector<int>   hI(nz), hJ(nz);
    std::vector<float> hval(nz), hX(N);

    cudaMemcpy(hI.data(),   d_I,   nz * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(hJ.data(),   d_J,   nz * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(hval.data(), d_val, nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hX.data(),   d_X,   N  * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "=== rank " << rank << " (nz=" << nz << ", N=" << N << ") ===\n";

    std::cout << "I   = [";
    for (int k = 0; k < nz; ++k) { std::cout << hI[k]; if (k+1 < nz) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "J   = [";
    for (int k = 0; k < nz; ++k) { std::cout << hJ[k]; if (k+1 < nz) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "val = [";
    for (int k = 0; k < nz; ++k) { std::cout << hval[k]; if (k+1 < nz) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "X   = [";
    for (int i = 0; i < N; ++i) { std::cout << hX[i]; if (i+1 < N) std::cout << ", "; }
    std::cout << "]\n";

    std::cout.flush();
}

void debug_print_device_csr(const int* d_O, const int* d_J,
                            const float* d_val, const float* d_X,
                            int M, int nz, int N, int P, int rank)
{
    std::vector<int>   hO(M + 1), hJ(nz);
    std::vector<float> hval(nz), hX(N);

    cudaMemcpy(hO.data(),   d_O,   (M + 1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(hJ.data(),   d_J,   nz * sizeof(int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(hval.data(), d_val, nz * sizeof(float),      cudaMemcpyDeviceToHost);
    cudaMemcpy(hX.data(),   d_X,   N  * sizeof(float),      cudaMemcpyDeviceToHost);

    std::cout << "=== rank " << rank << " (M=" << M << ", nz=" << nz
              << ", N=" << N << ") ===\n";

    std::cout << "O   = [";
    for (int r = 0; r <= M; ++r) { std::cout << hO[r]; if (r < M) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "J   = [";
    for (int k = 0; k < nz; ++k) { std::cout << hJ[k]; if (k+1 < nz) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "val = [";
    for (int k = 0; k < nz; ++k) { std::cout << hval[k]; if (k+1 < nz) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "X   = [";
    for (int i = 0; i < N; ++i) { std::cout << hX[i]; if (i+1 < N) std::cout << ", "; }
    std::cout << "]\n";

    std::cout << "rows:\n";
    for (int r = 0; r < M; ++r) {
        int g_row = r * P + rank;                 // local_to_global
        std::cout << "  l_row " << r << " (g_row " << g_row << "): ";
        if (hO[r] == hO[r + 1]) std::cout << "(empty)";
        for (int k = hO[r]; k < hO[r + 1]; ++k) {
            std::cout << "(" << hJ[k] << ", " << hval[k] << ")";
            if (k + 1 < hO[r + 1]) std::cout << " ";
        }
        std::cout << "\n";
    }

    std::cout.flush();
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

    #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    if (p == 0 && !MPIX_Query_cuda_support())
        fprintf(stderr, "WARNING: MPI is not CUDA-aware at runtime\n");
    #endif

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

    int M  = 0;
    int N  = 0;
    int nz = 0;

    int owned_N = 0;

    std::vector<int> I;
    std::vector<int> J;

    std::vector<float> val;
    std::vector<float> X;
    std::vector<float> Y;

    int * d_I = nullptr;
    int * d_J = nullptr;

    float * d_val = nullptr;
    float * d_X   = nullptr;
    float * d_Y   = nullptr;

    #pragma region loading_file

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

        X.resize(g_N);
        for (int i = 0; i < g_N; i++) X[i] =  static_cast<float>(genrand_real1());
    }

    CHECK_MPI(MPI_Bcast(&g_M, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&g_N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&g_nz, 1, MPI_INT, 0, MPI_COMM_WORLD));

    #pragma endregion loading_file

    // if (p == 0) print_coo_local(I, J, val, X, P, p);

    #pragma region partitionning_matrix

    M  = (g_M - p + P - 1) / P;
    N  = g_N;

    owned_N = (g_N - p + P - 1) / P;

    if (p == 0)
    {
        pack_by_rank(I, J, val, P);
        pack_x_by_rank(X, P);
    }

    nz = 0;
    if (p == 0)
    {
        std::vector<int> ccount(P, 0);

        for (int k = 0; k < g_nz; ++k)
            ccount[owner(I[k], P)]++;

        MPI_Scatter(ccount.data(), 1, MPI_INT, &nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(nullptr, 1, MPI_INT, &nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    #pragma endregion partitionning_matrix

    // if (p == 0) print_coo_local(I, J, val, X, P, p);

    #pragma region matrix_scatter

    if (p == 0)
    {
        cudaMalloc(&d_I,   g_nz * sizeof(int));
        cudaMalloc(&d_J,   g_nz * sizeof(int));
        cudaMalloc(&d_val, g_nz * sizeof(float));
        cudaMalloc(&d_X,   g_N  * sizeof(float));

        cudaMemcpy(d_I,   I.data(),   g_nz * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_J,   J.data(),   g_nz * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, val.data(), g_nz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X,   X.data(),   g_N  * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    scatter_in_place(g_nz, g_N, P, p, d_I, d_J, d_val, d_X, nz, owned_N);

    // debug_print_device(d_I, d_J, d_val, d_X, nz, owned_N, p);
    // MPI_Barrier(MPI_COMM_WORLD);

    #pragma endregion matrix_scatter

    #pragma region computing_csr
    int* d_O = nullptr;
    cudaMalloc(&d_O, (M + 1) * sizeof(int));
    cudaMemset(d_O, 0, (M + 1) * sizeof(int));

    int threads = 256, blocks = (nz + threads - 1) / threads;
    count_per_row<<<blocks, threads>>>(d_I, nz, P, d_O);

    thrust::device_ptr<int> t(d_O);
    thrust::inclusive_scan(t, t + (M + 1), t);

    debug_print_device_csr(d_O, d_J, d_val, d_X, M, nz, owned_N, P, p);
    MPI_Barrier(MPI_COMM_WORLD);

    #pragma endregion computing_csr

    #pragma region cleaning_up

    CHECK_CUDA(cudaFree(d_I));
    CHECK_CUDA(cudaFree(d_J));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_X));

    CHECK_MPI(MPI_Finalize());

    #pragma endregion cleaning_up

    return EXIT_SUCCESS;
}
