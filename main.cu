#pragma region includes

#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
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
#include "include/spmv_gpu.cuh"

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

// Lightweight CUDA-event timer for accurate device-side kernel timing.
struct GpuTimer
{
    cudaEvent_t a, b;
    GpuTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void  start()   { cudaEventRecord(a); }
    float stop_ms() {                                   // elapsed ms, blocks until done
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

// Warm up the MPI subsystem before timing. The first collective in a program
// pays one-time costs (connection setup, buffer pools, CUDA-aware/RDMA path
// init) that would otherwise be charged to whatever phase runs first. A few
// small dummy Allreduce + Barrier rounds prime those paths.
static void mpi_warmup(int reps)
{
    double scratch = 1.0, out = 0.0;
    for (int r = 0; r < reps; ++r) {
        MPI_Allreduce(&scratch, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

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

__global__ void count_per_row(const int* d_I, int nz, int P, int* d_O)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < nz)
    {
        int local_row = d_I[k] / P;
        atomicAdd(&d_O[local_row + 1], 1);
    }
}


#pragma endregion partitioning_helpers

#pragma region communication_helpers

__global__ void gather_owned(const float* d_X, const int* d_in_cols, float* d_out_vals, int n_incoming, int P)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n_incoming)
        d_out_vals[t] = d_X[d_in_cols[t] / P];   // owned => j/P
}

// Halo (ghost) exchange split into a one-time PLAN and a per-iteration APPLY.
//
// The sparsity pattern is fixed, so "who needs which remote x-entries" and the
// J remap are computed ONCE in halo_setup. Only the x VALUES change between
// SpMV iterations, so halo_apply (gather owned values -> Alltoallv -> place
// ghosts) is the repeatable per-SpMV communication that gets folded into the
// timed loop.
struct HaloPlan
{
    int P = 0;
    int owned_X = 0;
    int n_ghost = 0;        // remote x-entries this rank fetches  (recv volume)
    int n_incoming = 0;     // x-entries this rank serves to others (send volume)

    // host counts/displs for the per-iteration value Alltoallv
    std::vector<int> sendcounts, sdispls;   // our requests  (recv side of values)
    std::vector<int> recvcounts, rdispls;   // others' reqs  (send side of values)

    // device buffers reused every iteration
    int*   d_in_cols    = nullptr;   // size n_incoming: global cols others want
    float* d_out_vals   = nullptr;   // size n_incoming: gathered owned values to send
    float* d_ghost_vals = nullptr;   // size n_ghost   : received ghost values
};

// One-time setup: builds the exchange plan, remaps d_J to index the extended x,
// grows d_X to [owned ; ghosts], and allocates the reusable device buffers.
HaloPlan halo_setup(int* d_J, float*& d_X, int nz, int owned_X, int P, int rank)
{
    HaloPlan plan;
    plan.P = P;
    plan.owned_X = owned_X;

    // 1. columns to host
    std::vector<int> hJ(nz);
    cudaMemcpy(hJ.data(), d_J, nz * sizeof(int), cudaMemcpyDeviceToHost);

    // 2. distinct ghost columns (sort + unique, no associative container)
    std::vector<int> req_cols;
    req_cols.reserve(nz);
    for (int k = 0; k < nz; ++k) {
        int j = hJ[k];
        if (j % P != rank) req_cols.push_back(j);
    }
    std::sort(req_cols.begin(), req_cols.end());
    req_cols.erase(std::unique(req_cols.begin(), req_cols.end()), req_cols.end());

    // 3. per-owner sendcounts; re-bucket into owner-contiguous order for Alltoallv
    plan.sendcounts.assign(P, 0);
    plan.sdispls.assign(P, 0);
    for (int p = 0; p < (int)req_cols.size(); ++p) plan.sendcounts[req_cols[p] % P]++;
    for (int r = 1; r < P; ++r) plan.sdispls[r] = plan.sdispls[r-1] + plan.sendcounts[r-1];
    plan.n_ghost = (int)req_cols.size();

    std::vector<int> req_by_owner(plan.n_ghost);
    std::vector<int> slot_of_sorted(plan.n_ghost);   // appended slot, by sorted pos
    {
        std::vector<int> cursor = plan.sdispls;
        for (int p = 0; p < plan.n_ghost; ++p) {
            int j = req_cols[p];
            int dst = cursor[j % P]++;
            req_by_owner[dst] = j;
            slot_of_sorted[p] = owned_X + dst;
        }
    }

    // 4. tell each owner how many columns we want
    plan.recvcounts.assign(P, 0);
    plan.rdispls.assign(P, 0);
    MPI_Alltoall(plan.sendcounts.data(), 1, MPI_INT,
                 plan.recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < P; ++r) plan.rdispls[r] = plan.rdispls[r-1] + plan.recvcounts[r-1];
    plan.n_incoming = plan.rdispls[P-1] + plan.recvcounts[P-1];

    // 5. exchange the column lists (owner-grouped)
    std::vector<int> in_cols(plan.n_incoming);
    MPI_Alltoallv(req_by_owner.data(), plan.sendcounts.data(), plan.sdispls.data(), MPI_INT,
                  in_cols.data(),      plan.recvcounts.data(), plan.rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 6. allocate reusable device buffers; upload the fixed in_cols once
    cudaMalloc(&plan.d_in_cols,    (plan.n_incoming ? plan.n_incoming : 1) * sizeof(int));
    cudaMalloc(&plan.d_out_vals,   (plan.n_incoming ? plan.n_incoming : 1) * sizeof(float));
    cudaMalloc(&plan.d_ghost_vals, (plan.n_ghost    ? plan.n_ghost    : 1) * sizeof(float));
    if (plan.n_incoming > 0)
        cudaMemcpy(plan.d_in_cols, in_cols.data(), plan.n_incoming * sizeof(int),
                   cudaMemcpyHostToDevice);

    // 7. grow d_X = [owned ; ghosts] (ghost region filled each iteration)
    float* d_Xext = nullptr;
    cudaMalloc(&d_Xext, (owned_X + plan.n_ghost) * sizeof(float));
    cudaMemcpy(d_Xext, d_X, owned_X * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_X);
    d_X = d_Xext;

    // 8. remap J once to index the extended x directly
    for (int k = 0; k < nz; ++k) {
        int j = hJ[k];
        if (j % P == rank) {
            hJ[k] = j / P;                       // owned -> local slot
        } else {
            int lo = 0, hi = plan.n_ghost - 1, pos = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (req_cols[mid] == j) { pos = mid; break; }
                else if (req_cols[mid] < j) lo = mid + 1;
                else hi = mid - 1;
            }
            hJ[k] = slot_of_sorted[pos];         // ghost -> appended slot
        }
    }
    cudaMemcpy(d_J, hJ.data(), nz * sizeof(int), cudaMemcpyHostToDevice);

    return plan;
}

// Per-iteration value exchange: gather the owned x-values others requested,
// Alltoallv them, and drop the received ghosts into the tail of d_X. Reuses the
// plan's cached buffers and counts; this is the repeatable per-SpMV comm.
void halo_apply(HaloPlan& plan, float* d_X)
{
    if (plan.n_incoming > 0) {
        int threads = 256, blocks = (plan.n_incoming + threads - 1) / threads;
        gather_owned<<<blocks, threads>>>(d_X, plan.d_in_cols, plan.d_out_vals,
                                          plan.n_incoming, plan.P);
    }
    cudaDeviceSynchronize();   // gather must finish before MPI reads d_out_vals

    MPI_Alltoallv(plan.d_out_vals,   plan.recvcounts.data(), plan.rdispls.data(), MPI_FLOAT,
                  plan.d_ghost_vals, plan.sendcounts.data(), plan.sdispls.data(), MPI_FLOAT,
                  MPI_COMM_WORLD);

    if (plan.n_ghost > 0)
        cudaMemcpy(d_X + plan.owned_X, plan.d_ghost_vals, plan.n_ghost * sizeof(float),
                   cudaMemcpyDeviceToDevice);
}

void halo_free(HaloPlan& plan)
{
    cudaFree(plan.d_in_cols);
    cudaFree(plan.d_out_vals);
    cudaFree(plan.d_ghost_vals);
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

    // Benchmark settings shared by the SpMV and MPI timing.
    const int REPS    = 100;   // timed repetitions to average
    const int WARMUPS = 5;     // untimed warm-up rounds before timing

    // Prime the MPI subsystem so the first timed collective isn't penalized by
    // one-time setup costs.
    mpi_warmup(WARMUPS);

    #pragma endregion init

    #pragma region variables

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

    int* d_O = nullptr;

    // --- timing accumulators (milliseconds) ---
    double t_scatter    = 0.0;   // one-time: MPI matrix/vector scatter
    double t_halo_setup = 0.0;   // one-time: halo plan construction
    double t_gather     = 0.0;   // one-time: MPI result gather
    double t_halo_iter  = 0.0;   // per-SpMV: halo value exchange (breakdown)
    double t_spmv_iter  = 0.0;   // per-SpMV: local kernel          (breakdown)

    #pragma endregion variables

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

            if (mm_is_symmetric(matcode) && x != y)
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
        // for (int i = 0; i < g_N; i++) X[i] =  static_cast<float>(i + 1);
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

    // --- timed (one-time): MPI scatter phase ---
    mpi_warmup(WARMUPS);
    MPI_Barrier(MPI_COMM_WORLD);
    {
        double t0 = MPI_Wtime();
        scatter_in_place(g_nz, g_N, P, p, d_I, d_J, d_val, d_X, nz, owned_N);
        t_scatter = (MPI_Wtime() - t0) * 1000.0; // s -> ms
    }

    #pragma endregion matrix_scatter

    #pragma region computing_csr

    cudaMalloc(&d_O, (M + 1) * sizeof(int));
    cudaMemset(d_O, 0, (M + 1) * sizeof(int));

    int threads = 256, blocks = (nz + threads - 1) / threads;
    count_per_row<<<blocks, threads>>>(d_I, nz, P, d_O);

    thrust::device_ptr<int> t(d_O);
    thrust::inclusive_scan(t, t + (M + 1), t);

    #pragma endregion computing_csr

    #pragma region halo_setup

    // --- timed (one-time): halo plan construction ---
    HaloPlan plan;
    mpi_warmup(WARMUPS);
    MPI_Barrier(MPI_COMM_WORLD);
    {
        double t0 = MPI_Wtime();
        plan = halo_setup(d_J, d_X, nz, owned_N, P, p);
        t_halo_setup = (MPI_Wtime() - t0) * 1000.0; // s -> ms
    }
    const int n_ghost = plan.n_ghost;   // per-rank communication volume (recv)

    #pragma endregion halo_setup

    #pragma region local_spmv

    cudaMalloc(&d_Y, M * sizeof(float));
    cudaMemset(d_Y, 0, M * sizeof(float));

    threads = 128;                       // 4 warps/block
    int warps_needed = M;                // one warp per local row
    blocks = (warps_needed * 32 + threads - 1) / threads;
    size_t shmem = threads * sizeof(float);

    // Warm-up: run the fused (halo + SpMV) iteration a few times untimed so
    // caches, the MPI value path, and any one-time init are primed.
    for (int w = 0; w < WARMUPS; ++w) {
        halo_apply(plan, d_X);
        spmv_gpu_csr_opt<<<blocks, threads, shmem>>>(d_O, d_J, d_val, M, d_X, d_Y);
    }
    cudaDeviceSynchronize();

    // --- timed (per-SpMV): fused halo + kernel loop ---
    // The halo value exchange is folded into the SpMV loop (it is the true
    // per-iteration communication of a distributed SpMV). Halo and kernel are
    // accumulated separately so the comm/compute breakdown is available; their
    // sum is the per-SpMV execution time used for FLOP/s.
    {
        GpuTimer gt;
        double halo_accum_s = 0.0;   // seconds, per-rank
        double spmv_accum_s = 0.0;   // seconds, per-rank

        MPI_Barrier(MPI_COMM_WORLD);
        for (int rep = 0; rep < REPS; ++rep) {
            double th0 = MPI_Wtime();
            halo_apply(plan, d_X);
            halo_accum_s += (MPI_Wtime() - th0);

            gt.start();
            spmv_gpu_csr_opt<<<blocks, threads, shmem>>>(d_O, d_J, d_val, M, d_X, d_Y);
            spmv_accum_s += (double)gt.stop_ms() / 1000.0;
        }

        t_halo_iter = halo_accum_s / REPS * 1000.0;   // avg ms per SpMV
        t_spmv_iter = spmv_accum_s / REPS * 1000.0;   // avg ms per SpMV
    }

    #pragma endregion local_spmv

    #pragma region y_gather

    std::vector<int> ycount(P, 0), ydispl(P, 0);
    if (p == 0) {
        for (int r = 0; r < P; ++r) ycount[r] = (g_M - r + P - 1) / P;   // rows per rank
        for (int r = 1; r < P; ++r) ydispl[r] = ydispl[r-1] + ycount[r-1];
    }

    float* d_Yfull = nullptr;
    if (p == 0) cudaMalloc(&d_Yfull, g_M * sizeof(float));

    // --- timed (one-time): MPI gather phase ---
    mpi_warmup(WARMUPS);
    MPI_Barrier(MPI_COMM_WORLD);
    {
        double t0 = MPI_Wtime();
        MPI_Gatherv(d_Y,     M, MPI_FLOAT,
                    d_Yfull, ycount.data(), ydispl.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        t_gather = (MPI_Wtime() - t0) * 1000.0;  // s -> ms
    }

    if (p == 0) {
        std::vector<float> hYblock(g_M), hY(g_M);
        cudaMemcpy(hYblock.data(), d_Yfull, g_M * sizeof(float), cudaMemcpyDeviceToHost);

        for (int r = 0; r < P; ++r)
            for (int l = 0; l < ycount[r]; ++l)
                hY[l * P + r] = hYblock[ydispl[r] + l];   // block pos -> global row
        // hY is now y in global row order

        // NOTE: result output disabled during benchmarking so stdout carries
        //       only the CSV. Re-enable to dump the y vector for validation.
        // std::ostringstream oss;
        // for (int i = 0; i < g_M; ++i) oss << hY[i] << "\n";
        // std::cout << oss.str();
    }

    #pragma endregion y_gather

    #pragma region metrics_output

    // Per-phase wall-clock cost is the SLOWEST rank (a collective finishes only
    // when the last participant does), so reduce phase times with MPI_MAX.
    double max_scatter, max_halo_setup, max_gather, max_halo_iter, max_spmv_iter;
    MPI_Reduce(&t_scatter,    &max_scatter,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_halo_setup, &max_halo_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather,     &max_gather,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_halo_iter,  &max_halo_iter,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_spmv_iter,  &max_spmv_iter,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Per-rank load balance: NNZ per rank (min / avg / max). Sum of local nz is
    // g_nz, so avg = g_nz / P; reduce for the extremes.
    int nnz_min, nnz_max;
    MPI_Reduce(&nz, &nnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nz, &nnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Per-rank communication volume = ghost x-entries fetched (min / avg / max).
    int cv_min, cv_max;
    long long my_cv = (long long)n_ghost, cv_sum = 0;
    MPI_Reduce(&n_ghost, &cv_min, 1, MPI_INT,       MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&n_ghost, &cv_max, 1, MPI_INT,       MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_cv,   &cv_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (p == 0) {
        const double nnz_avg = (double)g_nz  / P;
        const double cv_avg  = (double)cv_sum / P;

        // Per-SpMV execution time = halo value exchange + local kernel.
        const double iter_ms = max_halo_iter + max_spmv_iter;

        // SpMV does 2*nnz flops (one multiply + one add per nonzero).
        const double iter_s        = iter_ms / 1000.0;
        const double spmv_s        = max_spmv_iter / 1000.0;
        const double gflops        = (iter_s > 0.0) ? (2.0 * (double)g_nz / iter_s / 1e9) : 0.0;
        const double gflops_kernel = (spmv_s > 0.0) ? (2.0 * (double)g_nz / spmv_s / 1e9) : 0.0;

        std::ostringstream oss;
        oss << "matrix,P,g_M,g_N,g_nz,"
               "nnz_min,nnz_avg,nnz_max,"
               "commvol_min,commvol_avg,commvol_max,"
               "scatter_ms,halo_setup_ms,gather_ms,"
               "halo_iter_ms,spmv_iter_ms,iter_ms,"
               "gflops,gflops_kernel\n";

        oss << argv[1] << ","
            << P << "," << g_M << "," << g_N << "," << g_nz << ","
            << nnz_min << "," << nnz_avg << "," << nnz_max << ","
            << cv_min  << "," << cv_avg  << "," << cv_max  << ",";
        oss << std::scientific << std::setprecision(6)
            << max_scatter << "," << max_halo_setup << "," << max_gather << ","
            << max_halo_iter << "," << max_spmv_iter << "," << iter_ms << ","
            << gflops << "," << gflops_kernel << "\n";

        std::cout << oss.str();
    }

    #pragma endregion metrics_output

    #pragma region cleaning_up

    halo_free(plan);

    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_I));
    CHECK_CUDA(cudaFree(d_J));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    if (p == 0) CHECK_CUDA(cudaFree(d_Yfull));

    CHECK_MPI(MPI_Finalize());

    #pragma endregion cleaning_up

    return EXIT_SUCCESS;
}
