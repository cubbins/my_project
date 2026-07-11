#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

const int MAX_BLOCK_SIZE = 64; // upper bound for block_size

// Kernel: one thread per query position t
// Implements one streaming block update:
//   - updates m[t], l[t]
//   - accumulates into o[t, :]
__global__ void flash_block_kernel(
    const float* __restrict__ q,      // [T, D]
    const float* __restrict__ k_blk,  // [B, D]  (B = end-start)
    const float* __restrict__ v_blk,  // [B, D]
    float* __restrict__ o,            // [T, D] (accumulator)
    float* __restrict__ m,            // [T]
    float* __restrict__ l,            // [T]
    int T, int D, int B, float scale)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    // local buffer for scores over this block
    float scores[MAX_BLOCK_SIZE];

    // 1) compute scores[t, i] for i in block
    float max_block = -1e30f;
    for (int i = 0; i < B; ++i) {
        float dot = 0.0f;
        const float* q_row = q + t * D;
        const float* k_row = k_blk + i * D;
        for (int d = 0; d < D; ++d) {
            dot += q_row[d] * k_row[d];
        }
        float s = dot * scale; // / sqrt(D)
        scores[i] = s;
        if (s > max_block) max_block = s;
    }

    // 2) update running max m[t]
    float m_old = m[t];
    float m_new = fmaxf(m_old, max_block);

    // 3) update l[t] (denominator accumulator)
    float l_old = l[t];
    float sum_exp_block = 0.0f;
    for (int i = 0; i < B; ++i) {
        sum_exp_block += expf(scores[i] - m_new);
    }
    float l_new = expf(m_old - m_new) * l_old + sum_exp_block;

    m[t] = m_new;
    l[t] = l_new;

    // 4) accumulate into o[t, :]
    float* o_row = o + t * D;
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int i = 0; i < B; ++i) {
            float w = expf(scores[i] - m_new); // weights for this block
            const float* v_row = v_blk + i * D;
            acc += w * v_row[d];
        }
        o_row[d] += acc;
    }
}

// Normalize: o[t, :] /= l[t]
__global__ void normalize_kernel(float* o, const float* l, int T, int D)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float denom = l[t];
    float* o_row = o + t * D;
    for (int d = 0; d < D; ++d) {
        o_row[d] /= denom;
    }
}

int main()
{
    // Sample sizes
    const int T = 8;   // sequence length
    const int D = 4;   // feature dimension
    const int block_size = 4; // streaming block size (must <= MAX_BLOCK_SIZE)

    if (block_size > MAX_BLOCK_SIZE) {
        fprintf(stderr, "block_size > MAX_BLOCK_SIZE\n");
        return EXIT_FAILURE;
    }

    float scale = 1.0f / std::sqrt((float)D);

    // Host allocations
    float h_q[T * D];
    float h_k[T * D];
    float h_v[T * D];

    // Simple deterministic sample data
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
            h_q[t * D + d] = 0.1f * (t + 1) * (d + 1);
            h_k[t * D + d] = 0.2f * (t + 1) * (d + 1);
            h_v[t * D + d] = 0.3f * (t + 1) * (d + 1);
        }
    }

    // Device allocations
    float *d_q, *d_k, *d_v, *d_o, *d_m, *d_l;
    CUDA_CHECK(cudaMalloc(&d_q, T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_l, T * sizeof(float)));

    // Copy q, k, v
    CUDA_CHECK(cudaMemcpy(d_q, h_q, T * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, T * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, T * D * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize o = 0, m = -1e9, l = 0
    CUDA_CHECK(cudaMemset(d_o, 0, T * D * sizeof(float)));

    float h_m_init[T];
    float h_l_init[T];
    for (int t = 0; t < T; ++t) {
        h_m_init[t] = -1e9f;
        h_l_init[t] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_m, h_m_init, T * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l, h_l_init, T * sizeof(float), cudaMemcpyHostToDevice));

    // Launch streaming over blocks along the key dimension
    int threads = 128;
    int blocks = (T + threads - 1) / threads;

    for (int start = 0; start < T; start += block_size) {
        int end = (start + block_size < T) ? (start + block_size) : T;
        int B = end - start;

        const float* d_k_blk = d_k + start * D;
        const float* d_v_blk = d_v + start * D;

        flash_block_kernel<<<blocks, threads>>>(
            d_q, d_k_blk, d_v_blk,
            d_o, d_m, d_l,
            T, D, B, scale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Normalize: o[t, :] /= l[t]
    normalize_kernel<<<blocks, threads>>>(d_o, d_l, T, D);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    float h_o[T * D];
    CUDA_CHECK(cudaMemcpy(h_o, d_o, T * D * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    printf("Output o (FlashAttention-like streaming softmax):\n");
    for (int t = 0; t < T; ++t) {
        printf("t=%d: ", t);
        for (int d = 0; d < D; ++d) {
            printf("%8.4f ", h_o[t * D + d]);
        }
        printf("\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_l));

    return 0;
}
