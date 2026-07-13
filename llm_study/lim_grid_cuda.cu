#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

struct SimConfig {
    int nx = 32;
    int ny = 32;
    //int steps = 5000;
    int steps = 5000*100;

    int sample_every = 20;

    double dt = 1e-13;

    double R_branch = 0.5;
    double L_branch = 1e-9;
    double C_node   = 1e-12;
    double G_node   = 1e-4;

    double inject_amp = 1e-3;
    int source_x = 0;
    int source_y = 0;

    int probe0_x = 0;
    int probe0_y = 0;
    int probe1_x = 15;
    int probe1_y = 15;
    //int probe2_x = 31;
    //int probe2_y = 31;
    int probe2_x = 63;
    int probe2_y = 63;

 

    std::string csv_path = "lim_waveforms.csv";
};

static int node_id(int x, int y, int nx) {
    return y * nx + x;
}

struct HostNetwork {
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

struct DeviceNetwork {
    int Nn = 0;
    int Nb = 0;

    int *d_branch_src = nullptr, *d_branch_dst = nullptr;
    double *d_R = nullptr, *d_L = nullptr, *d_E = nullptr;
    double *d_C = nullptr, *d_G = nullptr, *d_H = nullptr;
    int *d_node_branch_ptr = nullptr, *d_node_branch_ids = nullptr, *d_node_branch_signs = nullptr;

    double *d_V_a = nullptr, *d_V_b = nullptr;
    double *d_I_a = nullptr, *d_I_b = nullptr;
};

HostNetwork make_grid_network(const SimConfig& cfg) {
    HostNetwork net;
    net.Nn = cfg.nx * cfg.ny;

    net.C.assign(net.Nn, cfg.C_node);
    net.G.assign(net.Nn, cfg.G_node);
    net.H.assign(net.Nn, 0.0);

    int src_node = node_id(cfg.source_x, cfg.source_y, cfg.nx);
    if (src_node >= 0 && src_node < net.Nn) {
        net.H[src_node] = cfg.inject_amp;
    }

    struct Edge {
        int src;
        int dst;
    };
    std::vector<Edge> edges;
    edges.reserve((cfg.nx - 1) * cfg.ny + cfg.nx * (cfg.ny - 1));

    for (int y = 0; y < cfg.ny; ++y) {
        for (int x = 0; x < cfg.nx; ++x) {
            int a = node_id(x, y, cfg.nx);
            if (x + 1 < cfg.nx) {
                int b = node_id(x + 1, y, cfg.nx);
                edges.push_back({a, b});
            }
            if (y + 1 < cfg.ny) {
                int b = node_id(x, y + 1, cfg.nx);
                edges.push_back({a, b});
            }
        }
    }

    net.Nb = static_cast<int>(edges.size());
    net.branch_src.resize(net.Nb);
    net.branch_dst.resize(net.Nb);
    net.R.assign(net.Nb, cfg.R_branch);
    net.L.assign(net.Nb, cfg.L_branch);
    net.E.assign(net.Nb, 0.0);

    for (int b = 0; b < net.Nb; ++b) {
        net.branch_src[b] = edges[b].src;
        net.branch_dst[b] = edges[b].dst;
    }

    std::vector<std::vector<std::pair<int,int>>> adj(net.Nn);
    for (int b = 0; b < net.Nb; ++b) {
        int s = net.branch_src[b];
        int d = net.branch_dst[b];
        adj[s].push_back({b, +1});
        adj[d].push_back({b, -1});
    }

    net.node_branch_ptr.resize(net.Nn + 1);
    net.node_branch_ptr[0] = 0;
    for (int i = 0; i < net.Nn; ++i) {
        net.node_branch_ptr[i + 1] = net.node_branch_ptr[i] + static_cast<int>(adj[i].size());
    }

    int nnz = net.node_branch_ptr.back();
    net.node_branch_ids.resize(nnz);
    net.node_branch_signs.resize(nnz);

    for (int i = 0; i < net.Nn; ++i) {
        int base = net.node_branch_ptr[i];
        for (int k = 0; k < static_cast<int>(adj[i].size()); ++k) {
            net.node_branch_ids[base + k] = adj[i][k].first;
            net.node_branch_signs[base + k] = adj[i][k].second;
        }
    }

    return net;
}

void upload_network(DeviceNetwork& dev, const HostNetwork& host) {
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

void free_network(DeviceNetwork& dev) {
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

    double sumI = 0.0;
    int start = node_branch_ptr[i];
    int end   = node_branch_ptr[i + 1];

    for (int p = start; p < end; ++p) {
        int b = node_branch_ids[p];
        int s = node_branch_signs[p];
        sumI += static_cast<double>(s) * I_curr[b];
    }

    double numer = (C[i] / dt) * V_prev[i] + H[i] - sumI;
    double denom = (C[i] / dt) + G[i];
    V_next[i] = numer / denom;
}

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

static std::vector<double> download_vector(const double* d_ptr, int n) {
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

void write_csv_header(std::ofstream& ofs) {
    ofs << "step,time_s,V_probe0,V_probe1,V_probe2\n";
}

int main(int argc, char** argv) {
    SimConfig cfg;

    if (argc >= 3) {
        cfg.nx = std::atoi(argv[1]);
        cfg.ny = std::atoi(argv[2]);
        cfg.probe2_x = std::min(cfg.nx - 1, cfg.probe2_x);
        cfg.probe2_y = std::min(cfg.ny - 1, cfg.probe2_y);
        cfg.probe1_x = std::min(cfg.nx - 1, cfg.probe1_x);
        cfg.probe1_y = std::min(cfg.ny - 1, cfg.probe1_y);
    }

    HostNetwork host = make_grid_network(cfg);
    DeviceNetwork dev{};
    upload_network(dev, host);

    int p0 = node_id(cfg.probe0_x, cfg.probe0_y, cfg.nx);
    int p1 = node_id(cfg.probe1_x, cfg.probe1_y, cfg.nx);
    int p2 = node_id(cfg.probe2_x, cfg.probe2_y, cfg.nx);

    std::ofstream ofs(cfg.csv_path);
    if (!ofs) {
        std::cerr << "Failed to open CSV file: " << cfg.csv_path << "\n";
        free_network(dev);
        return 1;
    }
    write_csv_header(ofs);

    const int threads = 256;
    const int blocks_nodes = (dev.Nn + threads - 1) / threads;
    const int blocks_branches = (dev.Nb + threads - 1) / threads;

    double* V_prev = dev.d_V_a;
    double* V_next = dev.d_V_b;
    double* I_curr = dev.d_I_a;
    double* I_next = dev.d_I_b;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int n = 0; n < cfg.steps; ++n) {
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
            cfg.dt
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
            cfg.dt
        );
        CUDA_CHECK(cudaGetLastError());

        if ((n % cfg.sample_every) == 0) {
            std::vector<double> V_sample = download_vector(V_next, dev.Nn);
            ofs << n
                << "," << std::setprecision(16) << (n * cfg.dt)
                << "," << V_sample[p0]
                << "," << V_sample[p1]
                << "," << V_sample[p2]
                << "\n";
        }

        std::swap(V_prev, V_next);
        std::swap(I_curr, I_next);
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaDeviceSynchronize());

    auto V_final = download_vector(V_prev, dev.Nn);
    auto I_final = download_vector(I_curr, dev.Nb);

    std::cout << "LIM CUDA grid simulation complete\n";
    std::cout << "Grid: " << cfg.nx << " x " << cfg.ny << "\n";
    std::cout << "Nodes: " << dev.Nn << ", Branches: " << dev.Nb << "\n";
    std::cout << "Steps: " << cfg.steps << ", dt = " << cfg.dt << "\n";
    std::cout << "Elapsed kernel time: " << ms << " ms\n";
    std::cout << "CSV: " << cfg.csv_path << "\n";

    std::cout << "Final probe voltages:\n";
    std::cout << "  probe0 (" << cfg.probe0_x << "," << cfg.probe0_y << ") = " << V_final[p0] << "\n";
    std::cout << "  probe1 (" << cfg.probe1_x << "," << cfg.probe1_y << ") = " << V_final[p1] << "\n";
    std::cout << "  probe2 (" << cfg.probe2_x << "," << cfg.probe2_y << ") = " << V_final[p2] << "\n";

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    free_network(dev);
    return 0;
}
