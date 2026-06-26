#include <iostream>
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

#define LOCAL_INDEX(global_idx, rank, P) \
    (((global_idx) - (rank)) / (P))

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

__global__ void gatherXValues(float *toSend, float *xLocal, int *colsToServe, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) toSend[k] = xLocal[colsToServe[k]];
}

__global__ void scatterXValues(float *xForKernel, float *xGhost, int *flatRequest, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) xForKernel[flatRequest[k]] = xGhost[k];
}

// Average per-iteration comm time (ms) for THIS rank. CUDA-aware MPI_Alltoallv only.
double halo_comm_prof(float *d_toSend, float *d_xGhost,
                      const int *sendCounts, const int *sendDispls,
                      const int *recvCounts, const int *recvDispls,
                      MPI_Comm comm)
{
    // Warmup — same buffers, results discarded. First call is much slower.
    for (int k = 0; k < WARMUP; ++k)
        MPI_Alltoallv(d_toSend, sendCounts, sendDispls, MPI_FLOAT,
                      d_xGhost, recvCounts, recvDispls, MPI_FLOAT, comm);
    MPI_Barrier(comm);

    double sum = 0.0;
    for (int k = 0; k < ITERATION; ++k) {
        MPI_Barrier(comm);                 // align entry: time the collective, not skew
        double t0 = MPI_Wtime();
        MPI_Alltoallv(d_toSend, sendCounts, sendDispls, MPI_FLOAT,
                      d_xGhost, recvCounts, recvDispls, MPI_FLOAT, comm);
        double t1 = MPI_Wtime();
        sum += (t1 - t0);
    }
    return (sum / ITERATION) * 1e3;        // ms
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

    // std::cout << "rank=" << rank << "\n";
    // display_card_informations(dev_count);

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

    // Reading MTX file.
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

    std::vector<float> g_X(g_N);
    std::vector<float> g_Y(g_M);

    if (rank == 0)
    {
        for (int i = 0; i < g_N; i++) g_X[i] =  static_cast<float>(genrand_real1() * 100);
    }

    int nz = I.size();

    int M = 0;
    for (int r = rank; r < g_M; r += P) ++M;

    int N = 0;
    for (int r = rank; r < g_N; r += P) ++N;

    std::vector<float> Y(M);
    std::vector<float> X(N);

    if (rank == 0)
    {
        std::vector<int> sendcnt(P), disp(P);
        std::vector<float> cyclic_X;

        for (int i = 0; i < P; ++i)
        {
            for (int r = i; r < g_N; r += P) sendcnt[i]++;
        }

        disp[0] = 0;
        for (int r = 1; r < P; ++r)
            disp[r] = disp[r-1] + sendcnt[r-1];

        for (int r = 0; r < P; ++r)
        {
            for (int i = 0; i < g_N; ++i)
            {
                if (OWNER(i, P) == r) cyclic_X.push_back(g_X[i]);
            }
        }

        MPI_Scatterv(cyclic_X.data(), sendcnt.data(), disp.data(), MPI_FLOAT, X.data(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT, X.data(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    mm_sort_coo(I.data(), J.data(), val.data(), nz);





    fprintf(stderr, "[%d] dev=%d of %d\n", rank, local_dev, dev_count);
    #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    if (rank==0) fprintf(stderr, "cuda-aware runtime: %d\n", MPIX_Query_cuda_support());
    #endif




    // Phase 1
    // 1. Unique remote columns grouped by owner rank.
    std::vector<std::unordered_set<int>> needSet(P);
    for (int k = 0; k < nz; ++k) {
        int col = J[k];
        if (OWNER(col, P) != rank) needSet[OWNER(col, P)].insert(col);
    }

    // recvCols[s] = sorted unique global cols we need FROM rank s.
    std::vector<int> recvCounts(P, 0), recvDispls(P, 0);
    std::vector<int> recvColsFlat;                 // concatenated, in rank order
    for (int s = 0; s < P; ++s) {
        std::vector<int> v(needSet[s].begin(), needSet[s].end());
        std::sort(v.begin(), v.end());
        recvDispls[s] = (int)recvColsFlat.size();
        recvCounts[s] = (int)v.size();
        recvColsFlat.insert(recvColsFlat.end(), v.begin(), v.end());
    }
    int G = (int)recvColsFlat.size();              // total ghost entries

    // 2. After Alltoall, sendCounts[d] = how many of OUR entries rank d wants.
    std::vector<int> sendCounts(P, 0), sendDispls(P, 0);
    CHECK_MPI(MPI_Alltoall(recvCounts.data(), 1, MPI_INT,
                        sendCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    int S = 0;
    for (int d = 0; d < P; ++d) { sendDispls[d] = S; S += sendCounts[d]; }

    // 3. Exchange the global column-index lists.
    std::vector<int> sendColsFlat(S);
    CHECK_MPI(MPI_Alltoallv(recvColsFlat.data(), recvCounts.data(), recvDispls.data(), MPI_INT,
                            sendColsFlat.data(), sendCounts.data(), sendDispls.data(), MPI_INT,
                            MPI_COMM_WORLD));

    // 4. Convert served global columns -> LOCAL indices into our X. This is your colsToServe.
    std::vector<int> colsToServe(S);
    for (int k = 0; k < S; ++k)
        colsToServe[k] = LOCAL_INDEX(sendColsFlat[k], rank, P);

    std::unordered_map<int,int> colToSlot;
    colToSlot.reserve(2 * G);
    for (int g = 0; g < G; ++g) colToSlot[recvColsFlat[g]] = N + g;  // ghost slots

    std::vector<int> Jlocal(nz);
    for (int k = 0; k < nz; ++k) {
        int col = J[k];
        Jlocal[k] = (OWNER(col, P) == rank)
                ? LOCAL_INDEX(col, rank, P)   // local slot in [0, N)
                : colToSlot[col];             // ghost slot in [N, N+G)
    }

    // flatRequest = destination slots in xForKernel for received ghosts.
    std::vector<int> flatRequest(G);
    for (int g = 0; g < G; ++g) flatRequest[g] = N + g;

    // Phase 2 part 1
    float *d_xForKernel;                                   // N + G
    CHECK_CUDA(cudaMalloc(&d_xForKernel, (N + G) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_xForKernel, X.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int   *d_colsToServe, *d_flatRequest;
    float *d_toSend, *d_xGhost;
    CHECK_CUDA(cudaMalloc(&d_colsToServe, S * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_flatRequest, G * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_toSend,      S * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_xGhost,      G * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_colsToServe, colsToServe.data(), S*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_flatRequest, flatRequest.data(), G*sizeof(int), cudaMemcpyHostToDevice));

    // Phase 2 part 2
    auto grid = [](int n){ return (n + 255) / 256; };

    // (a) gather the values others requested, on device
    gatherXValues<<<grid(S), 256>>>(d_toSend, d_xForKernel, d_colsToServe, S);
    CHECK_CUDA(cudaDeviceSynchronize());          // MPI is NOT stream-ordered — sync before handing it the buffer

    double comm_ms = halo_comm_prof(d_toSend, d_xGhost,
                                sendCounts.data(), sendDispls.data(),
                                recvCounts.data(), recvDispls.data(),
                                MPI_COMM_WORLD);

    double comm_max = 0.0, comm_sum = 0.0;
    MPI_Reduce(&comm_ms, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_ms, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        fprintf(stderr, "halo comm: max=%.4f ms  avg=%.4f ms\n",
                comm_max, comm_sum / P);
    long sBytes = (long)S * sizeof(float);
    long rBytes = (long)G * sizeof(float);
    fprintf(stderr, "[%d] S=%d G=%d  send=%.2f MB recv=%.2f MB  eff=%.1f MB/s\n",
            rank, S, G, sBytes/1e6, rBytes/1e6,
            rBytes / (comm_ms * 1e-3) / 1e6);

    // A) device path — what you have
    MPI_Barrier(MPI_COMM_WORLD); double td0 = MPI_Wtime();
    MPI_Alltoallv(d_toSend, sendCounts.data(), sendDispls.data(), MPI_FLOAT,
                d_xGhost, recvCounts.data(), recvDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
    double td1 = MPI_Wtime();

    // B) host-staged path — copies included in the timed region
    std::vector<float> h_send(S), h_ghost(G);
    MPI_Barrier(MPI_COMM_WORLD); double th0 = MPI_Wtime();
    cudaMemcpy(h_send.data(), d_toSend, S*sizeof(float), cudaMemcpyDeviceToHost);
    MPI_Alltoallv(h_send.data(),  sendCounts.data(), sendDispls.data(), MPI_FLOAT,
                h_ghost.data(), recvCounts.data(), recvDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
    cudaMemcpy(d_xGhost, h_ghost.data(), G*sizeof(float), cudaMemcpyHostToDevice);
    double th1 = MPI_Wtime();

    fprintf(stderr, "[%d] device=%.3f ms  hoststaged=%.3f ms\n",
            rank, (td1-td0)*1e3, (th1-th0)*1e3);


    // (b) exchange GPU buffer -> GPU buffer directly (CUDA-aware MPI)
    CHECK_MPI(MPI_Alltoallv(d_toSend, sendCounts.data(), sendDispls.data(), MPI_FLOAT,
                            d_xGhost, recvCounts.data(), recvDispls.data(), MPI_FLOAT,
                            MPI_COMM_WORLD));

    // (c) scatter received ghosts into the kernel's X array, on device
    scatterXValues<<<grid(G), 256>>>(d_xForKernel, d_xGhost, d_flatRequest, G);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());          // before the SpMV kernel reads d_xForKernel (if on a different stream)

    // now launch your CSR SpMV using Jlocal and d_xForKernel






    std::vector<int> O(M + 1);
    for (int i = 0; i < nz; ++i) O[LOCAL_INDEX(I[i], rank, P) + 1]++;
    for (int i = 1; i <= M; ++i) O[i] += O[i - 1];

    MPI_Barrier(MPI_COMM_WORLD);
    // struct results caca = spmv_gpu_csr_opt_prof(O.data(), J.data(), val.data(), M, g_N, nz, NULL, g_X.data());
    // struct results caca = spmv_gpu_csr_opt_prof(O.data(), Jloc.data(), val.data(), M, N + total_recv, nz, NULL, d_X_ext);
    struct results caca = spmv_gpu_csr_opt_prof(O.data(), Jlocal.data(), val.data(), M, N + G, nz, NULL, d_xForKernel);

    // float *Xh = (float *)malloc((N + total_recv) * sizeof(float));
    // cudaMemcpy(Xh, d_X_ext, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CHECK_CUDA(cudaFree(d_X_ext));

    MPI_Barrier(MPI_COMM_WORLD);
    // CHECK_MPI(MPI_Bcast(g_X.data(), g_N, MPI_FLOAT, 0, MPI_COMM_WORLD));
    std::ostringstream oss;

    oss << "Rank Info:\n\n";
    oss << "   Rank . . . . . . . . . . . . . : " << rank << "\n";
    oss << "   M  . . . . . . . . . . . . . . : " << M << '\n';
    oss << "   N  . . . . . . . . . . . . . . : " << N + G << '\n';
    oss << "   NZ . . . . . . . . . . . . . . : " << nz << "\n\n";
    print_results(caca, argv[1], "csr", oss);
    // oss << "Comm Results:\n\n";
    // oss << "   Time . . . . . . . . . . . . . : " << t_halo * 1e3 << "\n\n";
    // oss << "Rank=" << rank << "\n";
    // for (int i = 0; i < nz; i++) oss << Jloc[i] << " ";
    // oss << "\n";
    // for (int i = 0; i < N + total_recv; i++) oss << Xh[i] << " ";
    // oss << "\n";
    // for (int i = 0; i < g_N; i++) oss << g_X[i] << " ";
    // oss << "\n";

    // free(Xh);

    // oss << "[" << rank << "] ";
    // for (int i = 0; i < 20; i++) oss << test[i] << " ";

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << oss.str() << std::endl;

    CHECK_MPI(MPI_Finalize());

    return 0;
}
