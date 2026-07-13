// sw_dpx_batch.cu — Advanced DPX Smith-Waterman: packed 16-bit SIMD batch alignment
// Targets sm_90a (Hopper H100) or sm_100a (Blackwell RTX 5060 / B-series)
//
// Compile (Hopper):
//   nvcc -O3 -arch=sm_90a -o sw_dpx_batch sw_dpx_batch.cu
//
// Compile (Blackwell RTX 5060):
//   nvcc -O3 -arch=sm_100a -o sw_dpx_batch sw_dpx_batch.cu
//
// Portable (runs on any GPU >= sm_70, uses DPX path on sm_90+):
//   nvcc -O3 -arch=sm_90 -code=sm_90,sm_100 -o sw_dpx_batch sw_dpx_batch.cu
//
// DESCRIPTION
// -----------
// Demonstrates two techniques from NVIDIA's DPX documentation:
//
//  1. dpx_sw_cell_packed() — device inline function showing the packed
//     short2 DPX intrinsic idiom (__vimax3_s16x2, __vimax_s16x2).
//     Two Smith-Waterman cells are computed simultaneously per register.
//
//  2. sw_dpx_batch() — a warp-parallel batch kernel where each CUDA
//     thread block aligns one query sequence against a reference using:
//       * Shared memory to cache the reference sequence
//       * Register-resident column vector (avoids global memory for H)
//       * DPX three-way max for the inner DP cell update
//       * Warp shuffle reduction to collect the maximum score
//
// Reference: "Boosting Dynamic Programming Performance Using NVIDIA Hopper
//   GPU DPX Instructions", NVIDIA Technical Blog, March 2023.
//   https://developer.nvidia.com/blog/boosting-dynamic-programming-...

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─── Constants ───────────────────────────────────────────────────────────────
#define MATCH_SCORE     2
#define MISMATCH_SCORE -1
#define GAP_PENALTY    -2
#define MAX_REF_LEN   4096
#define MAX_QUERY_LEN  512
#define THREADS_PER_BLOCK 128

// ════════════════════════════════════════════════════════════════════════════
//  PART 1 — Packed 16-bit DPX cell update (device helper)
// ════════════════════════════════════════════════════════════════════════════
//
// short2 holds TWO 16-bit signed integers { .x, .y }.
// On Hopper/Blackwell DPX instructions operate on BOTH lanes in a single
// instruction, effectively doubling the throughput of 32-bit operations.
//
// __vimax3_s16x2(a, b, c) :  result.x = max(a.x, b.x, c.x)
//                             result.y = max(a.y, b.y, c.y)
//
// __vimax_s16x2(a, b)      :  result.x = max(a.x, b.x)
//                             result.y = max(a.y, b.y)
//
// Note: __pack_s16x2(lo, hi) is a helper to pack two shorts into an int
//       for use with the raw PTX-mapped intrinsics.  The exact prototype
//       may vary by CUDA toolkit version; see cuda_runtime.h.

__device__ __forceinline__
void dpx_sw_cell_packed(
        short2&      H_cur,    // [out] result cell (two lanes)
        const short2 H_diag,   // H(i-1, j-1) for both lanes
        const short2 H_up,     // H(i-1, j)   for both lanes
        const short2 H_left,   // H(i,   j-1) for both lanes
        const short2 match_score) // +MATCH or +MISMATCH per lane
{
    // Step 1: diagonal score = H(i-1,j-1) + match/mismatch
    short2 diag_score;
    diag_score.x = (short)(H_diag.x + match_score.x);
    diag_score.y = (short)(H_diag.y + match_score.y);

    // Step 2: gap (affine gap model simplification — gap open only)
    short2 gap_up   = { (short)(H_up.x   + GAP_PENALTY),
                        (short)(H_up.y   + GAP_PENALTY) };
    short2 gap_left = { (short)(H_left.x + GAP_PENALTY),
                        (short)(H_left.y + GAP_PENALTY) };

    // Step 3: DPX three-way max across both packed lanes simultaneously
    //         128 operations per cycle per SM on Hopper / Blackwell
#if __CUDA_ARCH__ >= 900
    // Pack short2 fields into 32-bit words expected by DPX intrinsics
    int ds     = __pack_s16x2(diag_score.x, diag_score.y);
    int gu     = __pack_s16x2(gap_up.x,     gap_up.y);
    int gl     = __pack_s16x2(gap_left.x,   gap_left.y);
    int zero   = __pack_s16x2((short)0,     (short)0);

    int result = __vimax3_s16x2(ds, gu, gl);   // three-way max, both lanes
    result     = __vimax_s16x2(result, zero);   // clamp to 0 (local alignment)

    H_cur.x = (short)( result        & 0xFFFF);
    H_cur.y = (short)((result >> 16) & 0xFFFF);
#else
    // Fallback for pre-Hopper architectures
    H_cur.x = max((short)0,
                  max(diag_score.x, max(gap_up.x, gap_left.x)));
    H_cur.y = max((short)0,
                  max(diag_score.y, max(gap_up.y, gap_left.y)));
#endif
}

// ════════════════════════════════════════════════════════════════════════════
//  PART 2 — Batch alignment kernel (one query per thread block)
// ════════════════════════════════════════════════════════════════════════════
//
// Layout:
//   gridDim.x  = number of query sequences (one block per query)
//   blockDim.x = THREADS_PER_BLOCK (each thread handles one column stripe)
//
// The kernel avoids global memory for the DP column by keeping one
// "column score" register (H_prev) per thread.  Threads cooperate via
// shared memory (for the reference) and warp shuffles (for the reduction).
__global__
void sw_dpx_batch(
        const char* __restrict__ queries,   // [num_queries * query_len]
        const char* __restrict__ ref,       // [ref_len]
        int*        scores,                 // [num_queries] output
        int         query_len,
        int         ref_len)
{
    extern __shared__ short s_ref[];        // cached reference in shared mem

    const int qid = blockIdx.x;            // which query this block handles
    const int tid = threadIdx.x;

    // ── Load reference sequence into shared memory ──────────────────────
    for (int k = tid; k < ref_len; k += blockDim.x)
        s_ref[k] = (short)(unsigned char)ref[k];
    __syncthreads();

    const char* q = queries + (long long)qid * query_len;

    // ── DP loop ─────────────────────────────────────────────────────────
    // Each thread is responsible for column positions: tid, tid+blockDim.x,
    // tid+2*blockDim.x, ...  This strided approach keeps threads busy even
    // for short references, and the register H_prev accumulates state.
    short H_prev = 0;   // column score from previous row
    short H_max  = 0;   // running maximum for this query

    for (int i = 0; i < query_len; ++i) {
        short diag = 0, left = 0;

        for (int j = tid; j < ref_len; j += blockDim.x) {
            short s = (q[i] == (char)s_ref[j]) ? (short)MATCH_SCORE
                                                : (short)MISMATCH_SCORE;
            short cell_score = (short)(diag + s);

#if __CUDA_ARCH__ >= 900
            // DPX three-way max (32-bit version for scalar path)
            cell_score = (short)__vimax3_s32((int)cell_score,
                                             (int)(left   + GAP_PENALTY),
                                             (int)(H_prev + GAP_PENALTY));
            cell_score = (short)max((int)cell_score, 0);
#else
            cell_score = max((short)0,
                             max(cell_score,
                                 max((short)(left   + GAP_PENALTY),
                                     (short)(H_prev + GAP_PENALTY))));
#endif
            diag   = H_prev;
            H_prev = cell_score;
            left   = cell_score;

            if (cell_score > H_max) H_max = cell_score;
        }
        H_prev = 0;   // reset column for next query character
    }

    // ── Warp-level reduction to find maximum score across all threads ────
    // Each warp independently reduces; then thread 0 of each warp writes to
    // shared memory, and thread 0 does a final sweep.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        H_max = (short)max((int)H_max,
                           (int)__shfl_down_sync(0xFFFFFFFF, H_max, offset));

    // Shared memory reduction across warps
    __shared__ short warp_max[THREADS_PER_BLOCK / 32];
    if ((tid & (warpSize - 1)) == 0)
        warp_max[tid / warpSize] = H_max;
    __syncthreads();

    if (tid == 0) {
        short block_max = 0;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++)
            if (warp_max[w] > block_max) block_max = warp_max[w];
        scores[qid] = (int)block_max;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Host driver — aligns a small batch of queries against one reference
// ════════════════════════════════════════════════════════════════════════════
int main(void)
{
    // Example: three queries vs one reference
    const char *ref = "TACGTACGTACGGCTAGCTAGCTAGCTA";
    const char *query_strs[] = {
        "ACGTACGTAA",
        "GCTAGCTAGC",
        "TTTTGGGGCC"
    };
    const int NUM_QUERIES = 3;
    const int ref_len   = (int)strlen(ref);
    const int query_len = (int)strlen(query_strs[0]);  // all same length here

    printf("Reference (%d bp): %s\n\n", ref_len, ref);

    // Pack queries into a flat buffer
    char *h_queries = (char*)malloc(NUM_QUERIES * query_len);
    for (int q = 0; q < NUM_QUERIES; q++)
        memcpy(h_queries + q * query_len, query_strs[q], query_len);

    // ── Allocate device memory ─────────────────────────────────────────
    char *d_queries, *d_ref;
    int  *d_scores;
    cudaMalloc((void**)&d_queries, NUM_QUERIES * query_len * sizeof(char));
    cudaMalloc((void**)&d_ref,     ref_len * sizeof(char));
    cudaMalloc((void**)&d_scores,  NUM_QUERIES * sizeof(int));

    cudaMemcpy(d_queries, h_queries, NUM_QUERIES * query_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref,     ref,       ref_len,                 cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, NUM_QUERIES * sizeof(int));

    // ── Launch kernel ──────────────────────────────────────────────────
    // Shared memory = one short per reference base
    size_t smem = ref_len * sizeof(short);
    dim3 grid(NUM_QUERIES);
    dim3 block(THREADS_PER_BLOCK);

    sw_dpx_batch<<<grid, block, smem>>>(
        d_queries, d_ref, d_scores, query_len, ref_len);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // ── Retrieve and print results ─────────────────────────────────────
    int *h_scores = (int*)malloc(NUM_QUERIES * sizeof(int));
    cudaMemcpy(h_scores, d_scores, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Batch alignment results:\n");
    printf("%-25s  Score\n", "Query");
    printf("%-25s  -----\n", "-----");
    for (int q = 0; q < NUM_QUERIES; q++)
        printf("%-25s  %d\n", query_strs[q], h_scores[q]);

    // ── Cleanup ────────────────────────────────────────────────────────
    cudaFree(d_queries); cudaFree(d_ref); cudaFree(d_scores);
    free(h_queries); free(h_scores);
    return 0;
}
