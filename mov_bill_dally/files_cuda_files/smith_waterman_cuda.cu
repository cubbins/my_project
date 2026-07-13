// smith_waterman_cuda.cu — Illustrative CUDA Smith-Waterman
// Compile: nvcc -O3 -arch=sm_100 -o sw smith_waterman_cuda.cu
// Target:  RTX 5060 (Blackwell, compute capability 12.0)
//
// Also works on older GPUs (Ada, Ampere, Turing) via the #else path.
// For portable build covering both:
//   nvcc -O3 -arch=sm_90 -code=sm_90,sm_100 -o sw smith_waterman_cuda.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MATCH_SCORE    2
#define MISMATCH_SCORE -1
#define GAP_OPEN       -2
#define MAX_SEQ_LEN    1024

// ─── Scoring function ───────────────────────────────────────────────────────
__device__ __forceinline__
int score(char a, char b) {
    return (a == b) ? MATCH_SCORE : MISMATCH_SCORE;
}

// ─── Anti-diagonal wavefront kernel ────────────────────────────────────────
// Each thread handles one cell on the current anti-diagonal.
// This exploits the data dependency: H(i,j) depends on H(i-1,j-1),
// H(i-1,j), H(i,j-1) — cells on the PREVIOUS two anti-diagonals.
//
// Smith-Waterman recurrence:
//   H(i,j) = max(0,
//               H(i-1,j-1) + sigma(q_i, s_j),   <- match/mismatch
//               H(i-1,j)   + gap_open,            <- deletion
//               H(i,  j-1) + gap_open)            <- insertion
__global__
void sw_kernel(const char* __restrict__ query,
               const char* __restrict__ ref,
               int* H,         // scoring matrix (m+1) x (n+1), row-major
               int m, int n,   // sequence lengths
               int diag)       // current anti-diagonal index (2 .. m+n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Map anti-diagonal index + thread id -> (row i, col j)
    int i = tid + 1;
    int j = diag - tid;

    // Bounds check — threads outside the scoring matrix do nothing
    if (i < 1 || i > m || j < 1 || j > n) return;

    int up         = H[(i-1) * (n+1) + j]       + GAP_OPEN;
    int left       = H[ i    * (n+1) + (j-1)]   + GAP_OPEN;
    int diag_score = H[(i-1) * (n+1) + (j-1)]   + score(query[i-1], ref[j-1]);

    // DPX-style three-way max (intrinsic available on Hopper sm_90+ / Blackwell sm_100+)
    // H[i][j] = max(0, diag_score, up, left)
#if __CUDA_ARCH__ >= 900
    int val = __vimax3_s32(diag_score, up, left);
    H[i * (n+1) + j] = max(0, val);
#else
    H[i * (n+1) + j] = max(0, max(diag_score, max(up, left)));
#endif
}

// ─── Traceback helper (CPU) ─────────────────────────────────────────────────
// Finds the cell (max_i, max_j) with the highest score, then walks back
// diagonally to recover the aligned subsequences.
void traceback(const int* H, const char* query, const char* ref,
               int m, int n)
{
    // Locate maximum score
    int best = 0, bi = 0, bj = 0;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (H[i*(n+1)+j] > best) { best = H[i*(n+1)+j]; bi = i; bj = j; }

    printf("Best local alignment score : %d  (at row %d, col %d)\n", best, bi, bj);

    // Walk back
    char alnQ[MAX_SEQ_LEN*2] = {}, alnR[MAX_SEQ_LEN*2] = {}, mid[MAX_SEQ_LEN*2] = {};
    int len = 0, i = bi, j = bj;
    while (i > 0 && j > 0 && H[i*(n+1)+j] > 0) {
        int cur  = H[ i   *(n+1)+ j  ];
        int diag = H[(i-1)*(n+1)+(j-1)];
        int up   = H[(i-1)*(n+1)+ j  ];
        int left = H[ i   *(n+1)+(j-1)];
        if (cur == diag + score(query[i-1], ref[j-1])) {
            alnQ[len] = query[i-1];
            alnR[len] = ref[j-1];
            mid [len] = (query[i-1] == ref[j-1]) ? '|' : '.';
            i--; j--;
        } else if (cur == up + GAP_OPEN) {
            alnQ[len] = query[i-1]; alnR[len] = '-'; mid[len] = ' '; i--;
        } else {
            alnQ[len] = '-'; alnR[len] = ref[j-1]; mid[len] = ' '; j--;
        }
        len++;
    }
    // Reverse
    for (int k = 0; k < len/2; k++) {
        char tmp;
        tmp = alnQ[k]; alnQ[k] = alnQ[len-1-k]; alnQ[len-1-k] = tmp;
        tmp = alnR[k]; alnR[k] = alnR[len-1-k]; alnR[len-1-k] = tmp;
        tmp = mid [k]; mid [k] = mid [len-1-k];  mid [len-1-k] = tmp;
    }
    alnQ[len] = alnR[len] = mid[len] = '\0';
    printf("Query : %s\n", alnQ);
    printf("        %s\n", mid);
    printf("Ref   : %s\n", alnR);
}

// ─── Host driver ─────────────────────────────────────────────────────────────
int main(void)
{
    const char *query = "ACGTACGTAA";
    const char *ref   = "TACGTACGTA";
    int m = (int)strlen(query);
    int n = (int)strlen(ref);

    printf("Query : %s  (len=%d)\n", query, m);
    printf("Ref   : %s  (len=%d)\n\n", ref,   n);

    // Allocate host scoring matrix (zero-initialized)
    int *h_H = (int*)calloc((m+1)*(n+1), sizeof(int));

    // Allocate device memory
    int   *d_H;
    char  *d_q, *d_r;
    cudaMalloc((void**)&d_H, (m+1)*(n+1)*sizeof(int));
    cudaMalloc((void**)&d_q, m * sizeof(char));
    cudaMalloc((void**)&d_r, n * sizeof(char));

    cudaMemcpy(d_q, query, m * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, ref,   n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_H, 0, (m+1)*(n+1)*sizeof(int));

    // Launch one anti-diagonal per kernel call
    // Anti-diagonals run from index 2 (cell [1,1]) to m+n (cell [m,n])
    for (int diag = 2; diag <= m + n; ++diag) {
        // Number of active cells on this anti-diagonal
        int cells   = min(diag - 1, min(m, min(n, diag - 1)));
        int threads = min(cells, 256);
        int blocks  = (cells + threads - 1) / threads;
        sw_kernel<<<blocks, threads>>>(
            d_q, d_r, d_H, m, n, diag);
    }

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back and traceback
    cudaMemcpy(h_H, d_H, (m+1)*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);

    traceback(h_H, query, ref, m, n);

    cudaFree(d_H); cudaFree(d_q); cudaFree(d_r);
    free(h_H);
    return 0;
}
