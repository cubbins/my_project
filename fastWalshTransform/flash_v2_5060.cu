#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>

using namespace cuda::experimental;

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

// Tunable tile sizes
constexpr int TILE_Q = 64;   // queries per block
constexpr int TILE_K = 64;   // keys per tile
constexpr int D      = 64;   // head dimension (must match runtime D_actual for this demo)

// FlashAttention v2–style kernel: single head, no mask, batch=1
__global__ void flash_v2_kernel(
    const float* __restrict__ q,  // [T, D]
    const float* __restrict__ k,  // [T, D]
    const float* __restrict__ v,  // [T, D]
    float* __restrict__ o,        // [T, D]
    int T, int D_actual)
{
    extern __shared__ float smem[];
    float* smem_k = smem;                     // [TILE_K, D]
    float* smem_v = smem + TILE_K * D;        // [TILE_K, D]

    pipeline<thread_scope_thread> pipe = make_pipeline();

    const float scale = 1.0f / sqrtf((float)D_actual);

    int q_block_start = blockIdx.x * TILE_Q;
    int tid = threadIdx.x;

    // Each thread handles one query row within this block (up to TILE_Q)
    int q_idx = q_block_start + tid;
    bool valid_q = (tid < TILE_Q) && (q_idx < T);

    // Load Q row into registers
    float q_reg[D];
    if (valid_q) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            q_reg[d] = q[q_idx * D_actual + d];
        }
    }

    // Streaming softmax state
    float m_val = valid_q ? -1e9f : 0.0f;
    float l_val = valid_q ? 0.0f  : 1.0f;

    // Output accumulator in registers
    float o_reg[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        o_reg[d] = 0.0f;
    }

    int num_k_tiles = (T + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int k_start = tile * TILE_K;
        int B = min(TILE_K, T - k_start);

        // Async copy K and V tile into shared memory
        pipe.producer_acquire();

        // K tile
        for (int i = tid; i < B * D; i += blockDim.x) {
            smem_k[i] = k[k_start * D_actual + i];
        }
        // V tile
        for (int i = tid; i < B * D; i += blockDim.x) {
            smem_v[i] = v[k_start * D_actual + i];
        }

        pipe.producer_commit();
        pipe.consumer_wait();

        if (valid_q) {
            // 1) Compute scores for this tile: [B]
            float max_block = -1e30f;
            float scores[TILE_K]; // only first B used

            for (int i = 0; i < B; ++i) {
                const float* k_row = smem_k + i * D;
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    dot += q_reg[d] * k_row[d];
                }
                float s = dot * scale;
                scores[i] = s;
                if (s > max_block) max_block = s;
            }

            // 2) Update running max and l (online softmax)
            float m_new = fmaxf(m_val, max_block);

            float sum_exp = 0.0f;
            for (int i = 0; i < B; ++i) {
                sum_exp += __expf(scores[i] - m_new);
            }

            float l_new = __expf(m_val - m_new) * l_val + sum_exp;

            // 3) Update output accumulator
            //    o_new = exp(m_old - m_new) * o_old + sum_i exp(scores_i - m_new) * V_i
            float o_scale = __expf(m_val - m_new);

            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int i = 0; i < B; ++i) {
                    float w = __expf(scores[i] - m_new);
                    const float* v_row = smem_v + i * D;
                    acc += w * v_row[d];
                }
                o_reg[d] = o_scale * o_reg[d] + acc;
            }

            m_val = m_new;
            l_val = l_new;
        }

        pipe.consumer_release();
    }

    // Write back normalized output
    if (valid_q) {
        float inv_l = 1.0f / l_val;
        for (int d = 0; d < D; ++d) {
            o[q_idx * D_actual + d] = o_reg[d] * inv_l;
        }
    }
}

int main()
{
    const int T = 256;   // sequence length
    const int D_actual = D; // keep equal for this demo

    float *h_q = new float[T * D_actual];
    float *h_k = new float[T * D_actual];
    float *h_v = new float[T * D_actual];

    // Simple deterministic data
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D_actual; ++d) {
            h_q[t*D_actual + d] = 0.01f * (t + 1) * (d + 1);
            h_k[t*D_actual + d] = 0.02f * (t + 1) * (d + 1);
            h_v[t*D_actual + d] = 0.03f * (t + 1) * (d + 1);
        }
    }

    float *d_q, *d_k, *d_v, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, T * D_actual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, T * D_actual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, T * D_actual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, T * D_actual * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q, T * D_actual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, T * D_actual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, T * D_actual * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_Q);
    dim3 grid((T + TILE_Q - 1) / TILE_Q);
    size_t smem_bytes = 2 * TILE_K * D * sizeof(float); // K + V tiles

    flash_v2_kernel<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, T, D_actual);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_o = new float[T * D_actual];
    CUDA_CHECK(cudaMemcpy(h_o, d_o, T * D_actual * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first few rows
    printf("FlashAttention v2–style output (first 4 rows):\n");
    for (int t = 0; t < 4; ++t) {
        printf("t=%d: ", t);
        for (int d = 0; d < 8; ++d) {
            printf("%7.4f ", h_o[t*D_actual + d]);
        }
        printf("...\n");
    }

    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_o;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);

    return 0;
}
