// probe4.cu : verify the 1Cat frag->(row,col) expansion against the actual
// WMMA store layout on sm_75. One warp, one m16n16k16 QK tile.
// If the expansion matches hardware layout, the per-element mask marker
// applied at (row,col) lands at sS[row][col] after store_matrix_sync.
// Compile: nvcc -arch=sm_75 probe4.cu -o probe4 && ./probe4
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define S_STRIDE 32

__global__ void probe() {
    __shared__ half sQ[16 * S_STRIDE];
    __shared__ half sK[16 * S_STRIDE];
    __shared__ float sS[16 * S_STRIDE];

    const int lane = threadIdx.x % 32;

    // init smem: Q row-major 16x16 (row stride S_STRIDE), K col-major 16x16
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        const int r = i / 16, c = i % 16;
        // make Q = identity-ish and K = identity so acc[row][col] = 16 if row==col else 0
        sQ[r * S_STRIDE + c] = __float2half(r == c ? 1.0f : 0.0f);
        sK[c * S_STRIDE + r] = __float2half(r == c ? 1.0f : 0.0f); // K stored col-major, row=r, dim=c -> K[r][c]=I
    }
    __syncthreads();

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    load_matrix_sync(a_frag, sQ, S_STRIDE);                    // a = Q (16x16 identity)
    load_matrix_sync(b_frag, sK, S_STRIDE);                    // b = K (16x16 identity)
    fill_fragment(acc_frag, 0.0f);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 1Cat expansion
    const unsigned row_causal = (lane & 0b1) + ((lane >> 2) & 0b1) * 8 + ((lane >> 4) & 0b1) * 4;
    const unsigned col_causal = ((lane >> 1) & 0b1) * 2 + ((lane >> 3) & 0b1) * 8;

    // Apply a causal mask the way the kernel does: for each element, the
    // mask value at the expansion's (row,col) is added. Mask: col==0 -> 0,
    // else -> -inf. If the expansion matches the store layout, sS[r][0] keeps
    // the QK value and sS[r][c>0] becomes -inf.
    for (int i = 0; i < acc_frag.num_elements; ++i) {
        const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
        const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;
        (void) row;
        acc_frag.x[i] += (col == 0) ? 0.0f : -INFINITY;
    }
    store_matrix_sync(sS, acc_frag, S_STRIDE, mem_row_major);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("causal-mask applied via expansion, then stored:\n");
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const float v = sS[r * S_STRIDE + c];
                printf("  sS[%d][%d]= %s", r, c, (v < -1e20f) ? "-inf" : "val");
            }
            printf("\n");
        }
        printf("(expect every row: sS[r][0]=val, sS[r][1..3]=-inf)\n");
    }
}

int main() {
    probe<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}