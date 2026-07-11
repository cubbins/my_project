#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

constexpr int TILE_Q = 64;
constexpr int TILE_K = 64;
constexpr int D      = 64;

// Correct cp.async wrapper for a single float
__device__ __forceinline__
void cp_async_float(void* smem_ptr, const void* gmem_ptr)
{
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__
void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n");
}

__global__ void flash_v2_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int T, int D_actual)
{
    extern __shared__ float smem[];
    float* smem_k = smem;
    float* smem_v = smem + TILE_K * D;

    int q_block_start = blockIdx.x * TILE_Q;
    int tid = threadIdx.x;

    int q_idx = q_block_start + tid;
    bool valid_q = (tid < TILE_Q) && (q_idx < T);

    float q_reg[D];
    if (valid_q) {
        #pragma unroll
        for (int d = 0; d < D; d++)
            q_reg[d] = q[q_idx * D_actual + d];
    }

    float m_val = valid_q ? -1e9f : 0.0f;
    float l_val = valid_q ? 0.0f  : 1.0f;

    float o_reg[D] = {0};

    const float scale = 1.0f / sqrtf((float)D_actual);

    int num_tiles = (T + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int k_start = tile * TILE_K;
        int B = min(TILE_K, T - k_start);

        // Async copy K and V tiles
        for (int i = tid; i < B * D; i += blockDim.x) {
            cp_async_float(&smem_k[i], &k[k_start * D_actual + i]);
            cp_async_float(&smem_v[i], &v[k_start * D_actual + i]);
        }

        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        if (valid_q) {
            float scores[TILE_K];
            float max_block = -1e30f;

            for (int i = 0; i < B; i++) {
                const float* k_row = smem_k + i * D;
                float dot = 0.0f;

                #pragma unroll
                for (int d = 0; d < D; d++)
                    dot += q_reg[d] * k_row[d];

                float s = dot * scale;
                scores[i] = s;
                if (s > max_block) max_block = s;
            }

            float m_new = fmaxf(m_val, max_block);

            float sum_exp = 0.0f;
            for (int i = 0; i < B; i++)
                sum_exp += __expf(scores[i] - m_new);

            float l_new = __expf(m_val - m_new) * l_val + sum_exp;

            float o_scale = __expf(m_val - m_new);

            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int i = 0; i < B; i++) {
                    float w = __expf(scores[i] - m_new);
                    const float* v_row = smem_v + i * D;
                    acc += w * v_row[d];
                }
                o_reg[d] = o_scale * o_reg[d] + acc;
            }

            m_val = m_new;
            l_val = l_new;
        }

        __syncthreads();
    }

    if (valid_q) {
        float inv_l = 1.0f / l_val;
        for (int d = 0; d < D; d++)
            o[q_idx * D_actual + d] = o_reg[d] * inv_l;
    }
}

int main()
{
    const int T = 256;
    const int D_actual = D;

    float *h_q = new float[T * D_actual];
    float *h_k = new float[T * D_actual];
    float *h_v = new float[T * D_actual];

    for (int t = 0; t < T; t++)
        for (int d = 0; d < D_actual; d++) {
            h_q[t*D_actual + d] = 0.01f*(t+1)*(d+1);
            h_k[t*D_actual + d] = 0.02f*(t+1)*(d+1);
            h_v[t*D_actual + d] = 0.03f*(t+1)*(d+1);
        }

    float *d_q, *d_k, *d_v, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, T*D_actual*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, T*D_actual*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, T*D_actual*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, T*D_actual*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q, T*D_actual*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, T*D_actual*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, T*D_actual*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_Q);
    dim3 grid((T + TILE_Q - 1) / TILE_Q);
    size_t smem_bytes = 2 * TILE_K * D * sizeof(float);

    flash_v2_kernel<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, T, D_actual);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_o = new float[T * D_actual];
    CUDA_CHECK(cudaMemcpy(h_o, d_o, T*D_actual*sizeof(float), cudaMemcpyDeviceToHost));

    printf("FlashAttention v2 output (first 14 rows):\n");
    for (int t = 0; t < 14; t++) {
        printf("t=%d: ", t);
        for (int d = 0; d < 8; d++)
            printf("%7.4f ", h_o[t*D_actual + d]);
        printf("...\n");
    }

    return 0;
}
