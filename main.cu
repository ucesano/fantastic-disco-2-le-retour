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

    std::vector<float> X(N);
    std::vector<float> Y(M);

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














    // ---- Ghost columns: global cols we reference but don't own, grouped by owner ----
    std::unordered_set<int>       ghost_entries;     // unique ghost columns
    std::vector<std::vector<int>> need(P);           // need[p] = ghost cols owned by p

    for (int i = 0; i < nz; ++i)
    {
        int c = J[i];
        if (OWNER(c, P) != rank && ghost_entries.insert(c).second)
            need[OWNER(c, P)].push_back(c);
    }

    // recv_counts[p] = #values we want FROM p ; send_counts[p] = #values p wants FROM us
    std::vector<int> recv_counts(P), send_counts(P);
    for (int p = 0; p < P; ++p) recv_counts[p] = (int)need[p].size();
    CHECK_MPI(MPI_Alltoall(recv_counts.data(), 1, MPI_INT,
                           send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD));

    std::vector<int> recv_displs(P, 0), send_displs(P, 0);
    for (int p = 1; p < P; ++p)
    {
        recv_displs[p] = recv_displs[p-1] + recv_counts[p-1];
        send_displs[p] = send_displs[p-1] + send_counts[p-1];
    }
    int total_recv = recv_displs[P-1] + recv_counts[P-1];   // # ghosts we receive
    int total_send = send_displs[P-1] + send_counts[P-1];   // # values others want

    // Flatten our requests (global col ids) and swap them for the requests aimed at us.
    std::vector<int> recv_cols(total_recv), send_cols(total_send);
    for (int p = 0; p < P; ++p)
        std::copy(need[p].begin(), need[p].end(), recv_cols.begin() + recv_displs[p]);

    CHECK_MPI(MPI_Alltoallv(recv_cols.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                            send_cols.data(), send_counts.data(), send_displs.data(), MPI_INT,
                            MPI_COMM_WORLD));




    // // Pack the values others asked for, on the host: col c (we own it) -> X[(c-rank)/P]
    // std::vector<float> send_vals(total_send);
    // for (int t = 0; t < total_send; ++t)
    //     send_vals[t] = X[LOCAL_INDEX(send_cols[t], rank, P)];

    // // ---- Device extended X = [ owned (N) | ghosts (total_recv) ] ----
    // float *d_X_ext = nullptr;
    // float *d_send = nullptr;
    // CHECK_CUDA(cudaMalloc(&d_X_ext, (size_t)(N + total_recv) * sizeof(float)));
    // CHECK_CUDA(cudaMemcpy(d_X_ext, X.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // if (total_send > 0)
    // {
    //     CHECK_CUDA(cudaMalloc(&d_send, (size_t)total_send * sizeof(float)));
    //     CHECK_CUDA(cudaMemcpy(d_send, send_vals.data(),
    //                           (size_t)total_send * sizeof(float), cudaMemcpyHostToDevice));
    // }

    // // CUDA-aware: device send buffer -> device ghost region, no host staging.
    // CHECK_MPI(MPI_Alltoallv(d_send,      send_counts.data(), send_displs.data(), MPI_FLOAT,
    //                         d_X_ext + N, recv_counts.data(), recv_displs.data(), MPI_FLOAT,
    //                         MPI_COMM_WORLD));

    // Pack the values others asked for (unchanged) ...
    std::vector<float> send_vals(total_send);
    for (int t = 0; t < total_send; ++t)
        send_vals[t] = X[LOCAL_INDEX(send_cols[t], rank, P)];

    // Exchange actual values on the HOST — no CUDA-aware MPI needed.
    std::vector<float> recv_vals(total_recv);
    CHECK_MPI(MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(), MPI_FLOAT,
                            recv_vals.data(), recv_counts.data(), recv_displs.data(), MPI_FLOAT,
                            MPI_COMM_WORLD));

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Device extended X = [ owned (N) | ghosts (total_recv) ] ----
    float *d_X_ext = nullptr;
    CHECK_CUDA(cudaMalloc(&d_X_ext, (size_t)(N + total_recv) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X_ext, X.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    if (total_recv > 0)
        CHECK_CUDA(cudaMemcpy(d_X_ext + N, recv_vals.data(),
                            (size_t)total_recv * sizeof(float), cudaMemcpyHostToDevice));




    // const int COMM_WARMUP = 5;
    // const int COMM_ITER   = 50;

    // // warm up the path (first call sets up CUDA-aware staging / connections)
    // for (int k = 0; k < COMM_WARMUP; ++k)
    //     CHECK_MPI(MPI_Alltoallv(d_send,      send_counts.data(), send_displs.data(), MPI_FLOAT,
    //                             d_X_ext + N, recv_counts.data(), recv_displs.data(), MPI_FLOAT,
    //                             MPI_COMM_WORLD));

    // CHECK_CUDA(cudaDeviceSynchronize());   // GPU done before we start the clock
    // MPI_Barrier(MPI_COMM_WORLD);           // align ranks
    // double t0 = MPI_Wtime();

    // for (int k = 0; k < COMM_ITER; ++k)
    //     CHECK_MPI(MPI_Alltoallv(d_send,      send_counts.data(), send_displs.data(), MPI_FLOAT,
    //                             d_X_ext + N, recv_counts.data(), recv_displs.data(), MPI_FLOAT,
    //                             MPI_COMM_WORLD));

    // CHECK_CUDA(cudaDeviceSynchronize());   // ensure device-side completion
    // double t_halo = (MPI_Wtime() - t0) / COMM_ITER;

    // if (total_send > 0) CHECK_CUDA(cudaFree(d_send));

    // ---- Remap columns so SpMV indexes d_X_ext directly ----
    // owned col -> (c-rank)/P ; ghost col -> N + its slot in recv_cols
    std::unordered_map<int,int> ghost_pos;
    for (int t = 0; t < total_recv; ++t) ghost_pos[recv_cols[t]] = N + t;

    std::vector<int> Jloc(nz);
    for (int i = 0; i < nz; ++i)
        Jloc[i] = (OWNER(J[i], P) == rank) ? LOCAL_INDEX(J[i], rank, P)
                                           : ghost_pos[J[i]];

    // d_X_ext is now complete; SpMV reads d_X_ext[Jloc[i]].








    // CHECK_MPI(MPI_Bcast(g_X.data(), g_N, MPI_FLOAT, 0, MPI_COMM_WORLD));






    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> O(M + 1);
    for (int i = 0; i < nz; ++i) O[LOCAL_INDEX(I[i], rank, P) + 1]++;
    for (int i = 1; i <= M; ++i) O[i] += O[i - 1];

    // struct results caca = spmv_gpu_csr_opt_prof(O.data(), J.data(), val.data(), M, g_N, nz, NULL, g_X.data());
    // struct results caca = spmv_gpu_csr_opt_prof(O.data(), Jloc.data(), val.data(), M, N + total_recv, nz, NULL, d_X_ext);

    float *Xh = (float *)malloc((N + total_recv) * sizeof(float));
    cudaMemcpy(Xh, d_X_ext, N * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK_CUDA(cudaFree(d_X_ext));

    std::ostringstream oss;

    // oss << "Rank Info:\n\n";
    // oss << "   Rank . . . . . . . . . . . . . : " << rank << "\n";
    // oss << "   M  . . . . . . . . . . . . . . : " << M << '\n';
    // oss << "   N  . . . . . . . . . . . . . . : " << N + total_recv << '\n';
    // oss << "   NZ . . . . . . . . . . . . . . : " << nz << "\n\n";
    // print_results(caca, argv[1], "csr", oss);
    // oss << "Comm Results:\n\n";
    // oss << "   Time . . . . . . . . . . . . . : " << t_halo * 1e3 << "\n\n";
    oss << "Rank=" << rank << "\n";
    for (int i = 0; i < nz; i++) oss << Jloc[i] << " ";
    oss << "\n";
    for (int i = 0; i < N + total_recv; i++) oss << Xh[i] << " ";
    oss << "\n";
    for (int i = 0; i < g_N; i++) oss << g_X[i] << " ";
    oss << "\n";

    free(Xh);

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << oss.str() << std::endl;

    CHECK_MPI(MPI_Finalize());

    return 0;
}
