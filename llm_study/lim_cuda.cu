#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ============================================================
// Node update kernel
//
// V_next[i] = ((Ci/dt)*V_prev[i] + H[i] - sum_k sign_k*I[b_k]) / ((Ci/dt)+Gi)
//
// sign convention:
//   branch b has orientation src[b] -> dst[b]
//   I[b] > 0 means current flows src -> dst
//   for KCL sum(sign * I[b]):
//     +1 at src node
//     -1 at dst node
// ============================================================
__global__ void update_nodes_kernel(
    int Nn,
    const int* __restrict__ node_branch_ptr,
    const int* __restrict__ node_branch_ids,
    const int* __restrict__ node_branch_signs,
    const double* __restrict__ C,
    const double* __restrict__ G,
    const double* __restrict__ H,
    const double* __restrict__ I_curr,
    const double* __restrict__ V_prev,
    double* __restrict__ V_next,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nn) return;

    double kcl_sum = 0.0;

    int start = node_branch_ptr[i];
    int end   = node_branch_ptr[i + 1];

    for (int p = start; p < end; ++p) {
        int b = node_branch_ids[p];
        int s = node_branch_signs[p];
        kcl_sum += static_cast<double>(s) * I_curr[b];
    }

    double numer = (C[i] / dt) * V_prev[i] + H[i] - kcl_sum;
    double denom = (C[i] / dt) + G[i];

    V_next[i] = numer / denom;
}

// ============================================================
// Branch update kernel
//
// I_next[b] = I_curr[b] + (dt/L[b]) * (V[src]-V[dst]-R[b]*I_curr[b]+E[b])
// ============================================================
__global__ void update_branches_kernel(
    int Nb,
    const int* __restrict__ branch_src,
    const int* __restrict__ branch_dst,
    const double* __restrict__ R,
    const double* __restrict__ L,
    const double* __restrict__ E,
    const double* __restrict__ V_curr,
    const double* __restrict__ I_curr,
    double* __restrict__ I_next,
    double dt
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= Nb) return;

    int i = branch_src[b];
    int j = branch_dst[b];

    double vij = V_curr[i] - V_curr[j];
    double rhs = vij - R[b] * I_curr[b] + E[b];

    I_next[b] = I_curr[b] + (dt / L[b]) * rhs;
}

// ============================================================
// Utility: build a tiny demo network
//
// Example network:
//   node 0 --branch 0--> node 1 --branch 1--> node 2
//
// branch 0: 0 -> 1
// branch 1: 1 -> 2
//
// node incidence:
//   node 0: +b0
//   node 1: -b0, +b1
//   node 2: -b1
// ============================================================
struct LimNetworkHost {
    int Nn = 0;
    int Nb = 0;

    std::vector<int> branch_src;
    std::vector<int> branch_dst;

    std::vector<double> R;
    std::vector<double> L;
    std::vector<double> E;

    std::vector<double> C;
    std::vector<double> G;
    std::vector<double> H;

    std::vector<int> node_branch_ptr;
    std::vector<int> node_branch_ids;
    std::vector<int> node_branch_signs;
};

LimNetworkHost make_demo_network() {
    LimNetworkHost net;
    net.Nn = 3;
    net.Nb = 2;

    net.branch_src = {0, 1};
    net.branch_dst = {1, 2};

    net.R = {1.0, 1.0};
    net.L = {1e-9, 1e-9};
    net.E = {0.0, 0.0};

    net.C = {1e-12, 1e-12, 1e-12};
    net.G = {1e-3, 1e-3, 1e-3};

    // Inject a pulse/current into node 0
    net.H = {1e-3, 0.0, 0.0};

    // CSR incidence
    // node 0 -> [ b0(+1) ]
    // node 1 -> [ b0(-1), b1(+1) ]
    // node 2 -> [ b1(-1) ]
    net.node_branch_ptr   = {0, 1, 3, 4};
    net.node_branch_ids   = {0, 0, 1, 1};
    net.node_branch_signs = {+1, -1, +1, -1};

    return net;
}

// ============================================================
// Device container
// ============================================================
struct LimNetworkDevice {
    int Nn = 0;
    int Nb = 0;

    int* d_branch_src = nullptr;
    int* d_branch_dst = nullptr;

    double* d_R = nullptr;
    double* d_L = nullptr;
    double* d_E = nullptr;

    double* d_C = nullptr;
    double* d_G = nullptr;
    double* d_H = nullptr;

    int* d_node_branch_ptr = nullptr;
    int* d_node_branch_ids = nullptr;
    int* d_node_branch_signs = nullptr;

    double* d_V_a = nullptr;
    double* d_V_b = nullptr;

    double* d_I_a = nullptr;
    double* d_I_b = nullptr;
};

void allocate_and_copy(LimNetworkDevice& dev, const LimNetworkHost& host) {
    dev.Nn = host.Nn;
    dev.Nb = host.Nb;

    CUDA_CHECK(cudaMalloc(&dev.d_branch_src, host.Nb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.d_branch_dst, host.Nb * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&dev.d_R, host.Nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_L, host.Nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_E, host.Nb * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&dev.d_C, host.Nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_G, host.Nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_H, host.Nn * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&dev.d_node_branch_ptr, host.node_branch_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.d_node_branch_ids, host.node_branch_ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.d_node_branch_signs, host.node_branch_signs.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&dev.d_V_a, host.Nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_V_b, host.Nn * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&dev.d_I_a, host.Nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev.d_I_b, host.Nb * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dev.d_branch_src, host.branch_src.data(), host.Nb * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_branch_dst, host.branch_dst.data(), host.Nb * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev.d_R, host.R.data(), host.Nb * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_L, host.L.data(), host.Nb * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_E, host.E.data(), host.Nb * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev.d_C, host.C.data(), host.Nn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_G, host.G.data(), host.Nn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_H, host.H.data(), host.Nn * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev.d_node_branch_ptr, host.node_branch_ptr.data(),
                          host.node_branch_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_node_branch_ids, host.node_branch_ids.data(),
                          host.node_branch_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_node_branch_signs, host.node_branch_signs.data(),
                          host.node_branch_signs.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dev.d_V_a, 0, host.Nn * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev.d_V_b, 0, host.Nn * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev.d_I_a, 0, host.Nb * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev.d_I_b, 0, host.Nb * sizeof(double)));
}

void cleanup(LimNetworkDevice& dev) {
    cudaFree(dev.d_branch_src);
    cudaFree(dev.d_branch_dst);

    cudaFree(dev.d_R);
    cudaFree(dev.d_L);
    cudaFree(dev.d_E);

    cudaFree(dev.d_C);
    cudaFree(dev.d_G);
    cudaFree(dev.d_H);

    cudaFree(dev.d_node_branch_ptr);
    cudaFree(dev.d_node_branch_ids);
    cudaFree(dev.d_node_branch_signs);

    cudaFree(dev.d_V_a);
    cudaFree(dev.d_V_b);

    cudaFree(dev.d_I_a);
    cudaFree(dev.d_I_b);
}

// ============================================================
// Main simulation loop
// ============================================================
int main() {
    LimNetworkHost host = make_demo_network();
    LimNetworkDevice dev{};
    allocate_and_copy(dev, host);

    const double dt = 1e-13;
    const int steps = 1000;

    const int threads = 256;
    const int blocks_nodes = (dev.Nn + threads - 1) / threads;
    const int blocks_branches = (dev.Nb + threads - 1) / threads;

    double* V_prev = dev.d_V_a;
    double* V_next = dev.d_V_b;
    double* I_curr = dev.d_I_a;
    double* I_next = dev.d_I_b;

    for (int n = 0; n < steps; ++n) {
        update_nodes_kernel<<<blocks_nodes, threads>>>(
            dev.Nn,
            dev.d_node_branch_ptr,
            dev.d_node_branch_ids,
            dev.d_node_branch_signs,
            dev.d_C,
            dev.d_G,
            dev.d_H,
            I_curr,
            V_prev,
            V_next,
            dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_branches_kernel<<<blocks_branches, threads>>>(
            dev.Nb,
            dev.d_branch_src,
            dev.d_branch_dst,
            dev.d_R,
            dev.d_L,
            dev.d_E,
            V_next,
            I_curr,
            I_next,
            dt
        );
        CUDA_CHECK(cudaGetLastError());

        std::swap(V_prev, V_next);
        std::swap(I_curr, I_next);
    }

    std::vector<double> V_final(dev.Nn);
    std::vector<double> I_final(dev.Nb);

    CUDA_CHECK(cudaMemcpy(V_final.data(), V_prev, dev.Nn * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(I_final.data(), I_curr, dev.Nb * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Final node voltages:\n";
    for (int i = 0; i < dev.Nn; ++i) {
        std::cout << "  V[" << i << "] = " << V_final[i] << "\n";
    }

    std::cout << "Final branch currents:\n";
    for (int b = 0; b < dev.Nb; ++b) {
        std::cout << "  I[" << b << "] = " << I_final[b] << "\n";
    }

    cleanup(dev);
    return 0;
}
