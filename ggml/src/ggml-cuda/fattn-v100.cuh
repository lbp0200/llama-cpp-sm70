#pragma once
// Volta (SM70) / Turing (SM75) flash attention: 1Cat FLASH_ATTN_V100 dataflow
// ported to ggml.
//
// References:
// - 1Cat-vLLM flash-attention-v100/kernel/fused_mha_forward.cu (FlashAttention-1
//   style: QK scores materialized in fp32 smem, 32-col sub-tile softmax, PV WMMA
//   reading probabilities from smem; 16 warps / BLOCK_M=32 / BLOCK_N=64 for D=256).
// - llama.cpp #28037: Volta config tuning is near-optimal for the current fused
//   kernel design; the remaining headroom is architectural (split-D dataflow,
//   fp32 score smem, no cp.async => manual smem staging).
//
// v1 scope (locked for the first A/B):
// - CUDA only (no HIP/MUSA). cc == Volta (700) or Turing (750).
// - D = 256, fp16 K/V, fp32 Q input, fp32 output (llama.cpp FA conventions).
// - Causal / SWA via the f16 mask tensor (added after scale, like the VEC path).
// - Single sequence (ne[3] == 1). GQA handled in-kernel via gqa_ratio.
// - Env kill-switch: GGML_V100_FA=0 falls back to the existing kernels.
//
// Two configs:
// - cfg_v100: BLOCK_M=32 / BLOCK_N=64 / 512 threads (16 warps), ~92KB smem.
//   Volta/GV100 has 96KB smem per block, so it needs the raised limit.
// - cfg_sm75: BLOCK_M=16 / BLOCK_N=64 / 256 threads (8 warps), ~61KB smem.
//   Turing has 64KB smem per block and no cp.async either; the narrow M tile
//   keeps the same layout logic with double the blocks for the same work.

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <mma.h>

using namespace nvcuda::wmma;

namespace ggml_cuda_fattn_v100 {

// Debug dump (env GGML_V100_DEBUG=1). Zero-cost when unset: single static flag
// read per kernel entry, printf only from thread 0 of one block.
static bool v100_debug() {
    static const bool d = [] {
        const char * s = getenv("GGML_V100_DEBUG");
        return s && s[0] == '1';
    }();
    return d;
}

// WMMA tile shape used by 1Cat (validated against nvcuda::wmma by their harness).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

template <int BLOCK_M_, int BLOCK_N_, int THREADS_PER_BLOCK_>
struct cfg {
    static constexpr int BLOCK_M = BLOCK_M_;
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int THREADS_PER_BLOCK = THREADS_PER_BLOCK_;
    static constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    // THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M: 512/32 = 16 and 256/16 = 16.
    static constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;
    static constexpr int P_SUB_TILE = 32; // softmax sub-tile width (also a KV tile bound)

    static constexpr int D256_PAD = 0; // (8 - (256 % 32) + 32) % 32

    static constexpr int Q_STRIDE  = 256 + D256_PAD;
    static constexpr int KV_STRIDE = 256 + D256_PAD;
    static constexpr int S_STRIDE  = BLOCK_N + D256_PAD;
    static constexpr int P_STRIDE  = P_SUB_TILE + D256_PAD;
    static constexpr int O_STRIDE  = 256 + D256_PAD;

    static constexpr float NEG_INF = -1e30f;

    struct alignas(128) SmemLayout {
        alignas(16) half  q[BLOCK_M * Q_STRIDE];
        union {
            alignas(16) half k[BLOCK_N * KV_STRIDE];
            alignas(16) half v[BLOCK_N * KV_STRIDE];
        } kv;
        alignas(16) float s[BLOCK_M * S_STRIDE];
        alignas(16) half  p[BLOCK_M * P_STRIDE];
        alignas(16) float o[BLOCK_M * O_STRIDE];
        alignas(16) float row_max[BLOCK_M];
        alignas(16) float row_sum[BLOCK_M];
    };

    static constexpr size_t SMEM_BYTES = (sizeof(SmemLayout) + 127) & ~size_t(127);
};

// V100 (GV100, 96KB smem/block) and SM75 (TU116/TU106, 64KB smem/block) configs.
using cfg_v100 = cfg<32, 64, 512>;
using cfg_sm75 = cfg<16, 64, 256>;

template <typename CFG>
__device__ __forceinline__ void init_smem_v100(char * smem_raw) {
    constexpr int N_U4 = CFG::SMEM_BYTES / 16;
    const int tid = threadIdx.x;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
#pragma unroll 1
    for (int i = tid; i < N_U4; i += CFG::THREADS_PER_BLOCK) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(addr + (i << 4)),
                     "r"(0)
                     : "memory");
    }
    __syncthreads();
}

// Convert fp32 Q rows (llama.cpp graph convention) to fp16 smem tiles.
template <typename CFG>
__device__ __forceinline__ void load_q_f32_to_smem(
        const char * __restrict__ q_ptr, const int q_row_stride,
        half * __restrict__ sQ, const int valid_q_rows) {
    constexpr int D_H2 = 256 / 2; // half2 per row after conversion
    const int tid = threadIdx.x;
    const int total = valid_q_rows * D_H2;
    const int row_stride_u2 = q_row_stride / (2 * (int) sizeof(float));

    const float2 * q_vec = reinterpret_cast<const float2 *>(q_ptr);
    half2 * sQ_h2 = reinterpret_cast<half2 *>(sQ);

    for (int idx = tid; idx < total; idx += CFG::THREADS_PER_BLOCK) {
        const int row = idx / D_H2;
        const int vec = idx - row * D_H2;
        if (row < valid_q_rows) {
            const float2 f2 = __ldg(&q_vec[row * row_stride_u2 + vec]);
            sQ_h2[row * (CFG::Q_STRIDE / 2) + vec] = __floats2half2_rn(f2.x, f2.y);
        }
    }
}

// f16 K/V tiles from gmem into smem, uint4 vectorized __ldg loads.
template <typename CFG>
__device__ __forceinline__ void load_kv_f16_to_smem(
        const char * __restrict__ kv_ptr, const int64_t kv_row_stride,
        half * __restrict__ sKV, const int valid_k_rows) {
    constexpr int PER_UINT4 = 8; // halfs per 16-byte load
    const int tid = threadIdx.x;
    const int d_u4 = 256 / PER_UINT4;
    const int row_stride_u4 = (int) (kv_row_stride / (PER_UINT4 * (int) sizeof(half)));

    const uint4 * kv_vec = reinterpret_cast<const uint4 *>(kv_ptr);
    uint4 * sKV_vec = reinterpret_cast<uint4 *>(sKV);

    for (int idx = tid; idx < (valid_k_rows * d_u4); idx += CFG::THREADS_PER_BLOCK) {
        const int row = idx / d_u4;
        const int vec = idx - row * d_u4;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < valid_k_rows) {
            val = __ldg(&kv_vec[row * row_stride_u4 + vec]);
        }
        sKV_vec[row * (CFG::KV_STRIDE / PER_UINT4) + vec] = val;
    }
}

// ---------------------------------------------------------------------------
// Side kernel: per-query-row visibility ceiling (number of non-masked keys).
// One block per row; finds the last non -inf column of the f16 mask row. This
// bounds the main kernel's KV scan (causal/SWA skip) without in-kernel sync.
// A fully-visible mask yields kv_max[row] == N; a fully masked row yields 0.
// ---------------------------------------------------------------------------
__global__ static void fattn_v100_mask_kvmax_kernel(
        const half * __restrict__ mask, int * __restrict__ kv_max,
        const int N, const int M, const int64_t row_stride) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    const char * mrow = reinterpret_cast<const char *>(mask) + (int64_t) row * row_stride;
    int last = -1;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float mv = __half2float(__ldg(reinterpret_cast<const half *>(mrow) + i));
        if (mv != -INFINITY) {
            last = i; // ascending scan: later columns overwrite
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        last = max(last, __shfl_down_sync(0xFFFFFFFFu, last, off));
    }
    __shared__ int red[8]; // 256 threads / 32 per warp
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = last;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int c = -1;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            c = max(c, red[w]);
        }
        kv_max[row] = c >= 0 ? c + 1 : 0;
    }
}

} // namespace ggml_cuda_fattn_v100

// ---------------------------------------------------------------------------
// Device kernel. One block handles CFG::BLOCK_M query rows of one (batch, head)
// and walks the full causal KV range in CFG::BLOCK_N chunks.
// ---------------------------------------------------------------------------
template <typename CFG>
__global__ void __launch_bounds__(CFG::THREADS_PER_BLOCK, 2)
flash_attn_ext_v100_kernel(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const int * __restrict__ kv_max,
        const float * __restrict__ sinks_ptr,
        char       * __restrict__ dst_ptr,
        const int32_t M, const int32_t N,
        const int32_t H, const int32_t H_KV,
        const int32_t stride_Q1, const int64_t stride_Q2, const int64_t stride_Q3,
        const int32_t stride_K1, const int64_t stride_K2, const int64_t stride_K3,
        const int32_t stride_V1, const int64_t stride_V2, const int64_t stride_V3,
        const int32_t stride_M1, const int64_t stride_M2, const int64_t stride_M3,
        const int32_t stride_D1, const int64_t stride_D2, const int64_t stride_D3,
        const float scale,
        float      * __restrict__ debug_out,
        const bool   debug) {
    using namespace ggml_cuda_fattn_v100;
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D = 256;
    if (debug && threadIdx.x == 0) {
        printf("[v100] ENTER x=%d z=%d\n", blockIdx.x, blockIdx.z);
    }
    const int tid = threadIdx.x;

    const int batch_head_id = blockIdx.z;
    if (false && debug) {
        return; // DEBUG probe disabled
    }
    if (batch_head_id >= H) {
        return;
    }
    const int batch_id = batch_head_id / H;
    const int q_head_id = batch_head_id - batch_id * H;
    const int kv_group_size = H / H_KV;
    if (debug && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[v100] H=%d H_KV=%d gqa=%d N=%d M=%d x=%d z=%d\n",
            H, H_KV, kv_group_size, N, M, blockIdx.x, blockIdx.z);
    }
    if (kv_group_size == 0) {
        return; // defense: H < H_KV (group-ratio zero) must never divide by it
    }
    const int kv_head_id = q_head_id / kv_group_size;

    const int start_row = blockIdx.x * CFG::BLOCK_M;
    if (start_row >= M) {
        return;
    }

    const int valid_q_rows = min(CFG::BLOCK_M, M - start_row);

    // v2: block-level visibility ceiling from a precomputed kv_max array. The
    // ceiling of the block's last query row bounds how far this block scans;
    // rows that are narrower inside the block stay exact via the per-element
    // mask adds, so this is safe for any mask shape (causal, SWA, all-visible,
    // sink). kv_max == nullptr means no mask: scan the full range.
    int num_n_tiles = (N + CFG::BLOCK_N - 1) / CFG::BLOCK_N;
    const int kv_ceiling = (kv_max != nullptr) ? kv_max[start_row + valid_q_rows - 1] : -1;
    if (kv_ceiling >= 0) {
        num_n_tiles = min(num_n_tiles, (kv_ceiling + CFG::BLOCK_N - 1) / CFG::BLOCK_N);
    }

    const char * q_base = Q_ptr + stride_Q3 * batch_id + stride_Q2 * q_head_id;
    const char * k_base = K_ptr + stride_K3 * batch_id + stride_K2 * kv_head_id;
    const char * v_base = V_ptr + stride_V3 * batch_id + stride_V2 * kv_head_id;
    const char * m_base = mask_ptr + stride_M3 * batch_id + stride_M2 * 0;
    if (debug && threadIdx.x == 0) {
        printf("[v100] M=%d N=%d H=%d H_KV=%d gqa=%d qhead=%d kvhead=%d grid=(%d,%d,%d)\n",
            M, N, H, H_KV, kv_group_size, q_head_id, kv_head_id,
            blockIdx.x, blockIdx.y, blockIdx.z);
        printf("[v100] Q=%p K=%p V=%p m=%p d=%p\n",
            (void *) q_base, (void *) k_base, (void *) v_base,
            (void *) m_base, (void *) dst_ptr);
        printf("[v100] sK1=%d sK2=%lld sK3=%lld sV1=%d sV2=%lld sV3=%lld\n",
            stride_K1, stride_K2, stride_K3, stride_V1, stride_V2, stride_V3);
    }
    // FA output is permute(0,2,1,3): bytes map as [D][H][M][B], so the head
    // stride is nb[1] and the token stride is nb[2] (see ggml_flash_attn_ext).
    char *       d_base = dst_ptr + stride_D3 * batch_id;
    if (kv_ceiling == 0) {
        // no visible key for any row in this block: zero the output and return
        const int n4 = valid_q_rows * (D / 4);
        for (int i = tid; i < n4; i += CFG::THREADS_PER_BLOCK) {
            const int row = i / (D / 4);
            const int col = (i - row * (D / 4)) * 4;
            *reinterpret_cast<float4 *>(d_base
                + (int64_t) (start_row + row) * stride_D2
                + (int64_t) q_head_id * stride_D1
                + (int64_t) col * sizeof(float)) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        return;
    }

    extern __shared__ char smem_raw[];
    init_smem_v100<CFG>(smem_raw);
    typename CFG::SmemLayout & smem = *reinterpret_cast<typename CFG::SmemLayout *>(smem_raw);

    half  * sQ = smem.q;
    half  * sK = smem.kv.k;
    half  * sV = smem.kv.v;
    float * sS = smem.s;
    half  * sP = smem.p;
    float * sO = smem.o;
    float * sRowMax = smem.row_max;
    float * sRowSum = smem.row_sum;

    if (threadIdx.x < CFG::BLOCK_M) {
        sRowMax[threadIdx.x] = CFG::NEG_INF;
    }

    load_q_f32_to_smem<CFG>(q_base + (int64_t) start_row * stride_Q1, stride_Q1, sQ, valid_q_rows);
    __syncthreads();

    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * CFG::BLOCK_N;
        const int valid_k_rows = min(CFG::BLOCK_N, N - start_col);

        // QK: load K tile, warps x m16n16k16 over D, store fp32 scores to smem.
        load_kv_f16_to_smem<CFG>(k_base + (int64_t) start_col * stride_K1, stride_K1, sK, valid_k_rows);
        __syncthreads();

        const int num_tiles_m_qk = (CFG::BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_qk = (CFG::BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_qk = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_qk = num_tiles_m_qk * num_tiles_n_qk;
        const int tiles_per_warp_qk = (total_tiles_qk + CFG::WARPS_PER_BLOCK - 1) / CFG::WARPS_PER_BLOCK;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        // nvcuda::wmma m16n16k16 fragment -> (row, col) permute, from 1Cat.

        for (int tile_idx = 0; tile_idx < tiles_per_warp_qk; ++tile_idx) {
            const int global_tile_idx = warp_id * tiles_per_warp_qk + tile_idx;
            if (global_tile_idx >= total_tiles_qk) {
                break;
            }

            const int tile_m_idx = global_tile_idx / num_tiles_n_qk;
            const int tile_n_idx = global_tile_idx - tile_m_idx * num_tiles_n_qk;
            const int tile_m = tile_m_idx * WMMA_M;
            const int tile_n = tile_n_idx * WMMA_N;
            if (tile_m >= valid_q_rows || tile_n >= valid_k_rows) {
                continue;
            }

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            fill_fragment(acc_frag, 0.0f);

#pragma unroll
            for (int k_tile = 0; k_tile < num_tiles_k_qk; ++k_tile) {
                const int k_off = k_tile * WMMA_K;
                if (k_off >= D) {
                    break;
                }
                load_matrix_sync(a_frag, sQ + tile_m * CFG::Q_STRIDE + k_off, CFG::Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * CFG::KV_STRIDE + k_off, CFG::KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

#pragma unroll
            // Raw scores only: scale + mask are applied after the store on the
            // smem tile by explicit (row, col). The WMMA fragment element order
            // differs between Volta (HMMA.884) and Turing (HMMA.16816), so a
            // fixed acc_frag.x[i] -> (row, col) expansion is not portable.
            store_matrix_sync(sS + tile_m * CFG::S_STRIDE + tile_n, acc_frag, CFG::S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // Scale + mask by explicit (row, col), architecture independent.
        for (int i = tid; i < valid_q_rows * CFG::BLOCK_N; i += CFG::THREADS_PER_BLOCK) {
            const int row = i / CFG::BLOCK_N;
            const int col = i - row * CFG::BLOCK_N;
            const int global_m = start_row + row;
            const int global_n = start_col + col;
            float v;
            if (global_n < start_col + valid_k_rows) {
                const half mask_val = __ldg(reinterpret_cast<const half *>(m_base + (int64_t) global_m * stride_M1) + global_n);
                v = sS[row * CFG::S_STRIDE + col] * scale + __half2float(mask_val);
            } else {
                v = CFG::NEG_INF;
            }
            sS[row * CFG::S_STRIDE + col] = v;
        }
        __syncthreads();
        if (debug_out != nullptr && debug && threadIdx.x == 0 && blockIdx.z == 1 && blockIdx.x == 0) {
            {
                int v0 = 0, v1 = 0;
                for (int c = 0; c < 32; ++c) {
                    if (sS[c] > -1e20f) v0++;
                    if (sS[CFG::S_STRIDE + c] > -1e20f) v1++;
                }
                debug_out[0] = (float) v0;
                debug_out[1] = (float) v1;
                debug_out[2] = (float) sS[0];
                debug_out[3] = (float) sS[16];
                debug_out[4] = (float) N;
                debug_out[5] = (float) M;
                debug_out[6] = (float) start_row;
                debug_out[7] = (float) valid_q_rows;
                for (int cc = 0; cc < 4; ++cc) {
                    debug_out[96 + cc] = __half2float(__ldg(reinterpret_cast<const half *>(m_base + (int64_t) 0 * stride_M1) + cc)); // direct mask[0][cc]
                }
            }
        }
        // PV: load V into the K/V union slot, sub-tile softmax, WMMA PV.
        load_kv_f16_to_smem<CFG>(v_base + (int64_t) start_col * stride_V1, stride_V1, sV, valid_k_rows);
        __syncthreads();

        const int num_tiles_m_pv = (CFG::BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_d_pv = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_d_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + CFG::WARPS_PER_BLOCK - 1) / CFG::WARPS_PER_BLOCK;
        const int p_tile_capacity = CFG::P_SUB_TILE;

        for (int sub_start = 0; sub_start < valid_k_rows; sub_start += p_tile_capacity) {
            const int sub_valid_k_rows = min(p_tile_capacity, valid_k_rows - sub_start);

            if (tid < valid_q_rows * CFG::THREADS_PER_ROW) {
                const int row = tid / CFG::THREADS_PER_ROW;
                const int thread_in_row = tid - row * CFG::THREADS_PER_ROW;
                const unsigned sync_mask =
                    (valid_q_rows == CFG::BLOCK_M) ? 0xFFFFFFFFU : __activemask();
                const int row_leader = __ffs(sync_mask) - 1;

                float * sS_row_f = sS + row * CFG::S_STRIDE + sub_start;
                half  * sP_row_h = sP + row * CFG::P_STRIDE;

                const int vec_cols = sub_valid_k_rows >> 2;
                const int vecs_per_thread = (vec_cols + CFG::THREADS_PER_ROW - 1) / CFG::THREADS_PER_ROW;
                const int tail_start = vec_cols << 2;

                float thread_max = CFG::NEG_INF;
                float4 * sS_vec4 = reinterpret_cast<float4 *>(sS_row_f);

#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j) {
                    const int vc = thread_in_row + j * CFG::THREADS_PER_ROW;
                    if (vc < vec_cols) {
                        const float4 v4 = sS_vec4[vc];
                        thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                    }
                }
#pragma unroll 4
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows; c += CFG::THREADS_PER_ROW) {
                    thread_max = fmaxf(thread_max, sS_row_f[c]);
                }
#pragma unroll
                for (int o = CFG::THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_max = fmaxf(thread_max, __shfl_down_sync(sync_mask, thread_max, o, CFG::THREADS_PER_ROW));
                }

                const float row_max = __shfl_sync(sync_mask, thread_max, row_leader, CFG::THREADS_PER_ROW);
                const float old_max = sRowMax[row];
                const float new_max = fmaxf(old_max, row_max);
                // A fully-masked row (max stays -inf) must not feed exp(NaN) via
                // (-inf - -inf). Clamp the exponent base to 0: exp(-inf)=0 keeps
                // the running sum at zero and the final output becomes exactly 0.
                const float safe_max = (new_max > -1e30f) ? new_max : 0.0f;
                const float exp_diff = expf(fmaxf(old_max - safe_max, -80.0f));

                float thread_sum = 0.0f;
                half2 half_buffer[20];
                int vc_base = thread_in_row;
                int h2_idx = 0;
                int tail_col = -1;
                half tail_value = __float2half(0.f);

#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j, vc_base += CFG::THREADS_PER_ROW) {
                    if (vc_base < vec_cols) {
                        const float4 v4 = sS_vec4[vc_base];
                        const float e0 = expf(fmaxf(v4.x - safe_max, -80.0f));
                        const float e1 = expf(fmaxf(v4.y - safe_max, -80.0f));
                        const float e2 = expf(fmaxf(v4.z - safe_max, -80.0f));
                        const float e3 = expf(fmaxf(v4.w - safe_max, -80.0f));
                        thread_sum += (e0 + e1) + (e2 + e3);
                        half_buffer[h2_idx++] = __float22half2_rn(make_float2(e0, e1));
                        half_buffer[h2_idx++] = __float22half2_rn(make_float2(e2, e3));
                    }
                }
#pragma unroll 4
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows; c += CFG::THREADS_PER_ROW) {
                    const float e = expf(fmaxf(sS_row_f[c] - safe_max, -80.0f));
                    thread_sum += e;
                    tail_col = c;
                    tail_value = __float2half_rn(e);
                }
#pragma unroll
                for (int o = CFG::THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_sum += __shfl_down_sync(sync_mask, thread_sum, o, CFG::THREADS_PER_ROW);
                }

                const float row_sum = __shfl_sync(sync_mask, thread_sum, row_leader, CFG::THREADS_PER_ROW);
                if (thread_in_row == 0) {
                    sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                    sRowMax[row] = new_max;
                }

                h2_idx = 0;
                vc_base = thread_in_row;
                half2 * sP_half2 = reinterpret_cast<half2 *>(sP_row_h);
#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j, vc_base += CFG::THREADS_PER_ROW) {
                    if (vc_base < vec_cols) {
                        const int base_offset = vc_base * 2;
                        sP_half2[base_offset] = half_buffer[h2_idx++];
                        sP_half2[base_offset + 1] = half_buffer[h2_idx++];
                    }
                }
                if (tail_col >= 0) {
                    sP_row_h[tail_col] = tail_value;
                }
#pragma unroll 4
                for (int c = tail_start + thread_in_row; c < p_tile_capacity; c += CFG::THREADS_PER_ROW) {
                    if (c >= sub_valid_k_rows) {
                        sP_row_h[c] = __float2half(0.f);
                    }
                }

                if (block_n > 0 || sub_start > 0) {
                    float * sO_row = sO + row * CFG::O_STRIDE;
                    float4 * sO_vec = reinterpret_cast<float4 *>(sO_row);
                    const int o_vec_count = (CFG::O_STRIDE + 3) >> 2;
#pragma unroll 4
                    for (int ov = thread_in_row; ov < o_vec_count; ov += CFG::THREADS_PER_ROW) {
                        float4 v = sO_vec[ov];
                        v.x *= exp_diff;
                        v.y *= exp_diff;
                        v.z *= exp_diff;
                        v.w *= exp_diff;
                        sO_vec[ov] = v;
                    }
                }
            }
            __syncthreads();
            if (debug_out != nullptr && debug && threadIdx.x == 0 && blockIdx.z == 1 && blockIdx.x == 0) {
                for (int c = 0; c < 8; ++c) {
                    debug_out[40 + c] = __half2float(sP[c]);
                }
                debug_out[48] = sRowMax[0];
                debug_out[49] = sRowSum[0];
            }
            const int num_tiles_k_pv = (p_tile_capacity + WMMA_K - 1) / WMMA_K;

            for (int tile_idx = 0; tile_idx < tiles_per_warp_pv; ++tile_idx) {
                const int global_tile_idx = warp_id * tiles_per_warp_pv + tile_idx;
                if (global_tile_idx >= total_tiles_pv) {
                    break;
                }

                const int tile_m_idx = global_tile_idx / num_tiles_d_pv;
                const int tile_d_idx = global_tile_idx - tile_m_idx * num_tiles_d_pv;
                const int tile_m = tile_m_idx * WMMA_M;
                const int tile_d = tile_d_idx * WMMA_N;
                if (tile_m >= valid_q_rows) {
                    continue;
                }

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

                load_matrix_sync(acc_frag, sO + tile_m * CFG::O_STRIDE + tile_d, CFG::O_STRIDE, mem_row_major);

#pragma unroll
                for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                    const int k_off = tile_k * WMMA_K;
                    if (k_off >= sub_valid_k_rows) {
                        break;
                    }
                    load_matrix_sync(a_frag, sP + tile_m * CFG::P_STRIDE + k_off, CFG::P_STRIDE);
                    load_matrix_sync(b_frag, sV + (sub_start + k_off) * CFG::KV_STRIDE + tile_d, CFG::KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sO + tile_m * CFG::O_STRIDE + tile_d, acc_frag, CFG::O_STRIDE, mem_row_major);
            }
            __syncthreads();
        }
    }

    // Sinks: an extra no-V softmax term per head. Same math as the VEC kernels:
    // max/sum absorb exp(sink), and the accumulated O is rescaled by exp_diff.
    if (sinks_ptr != nullptr) {
        const float sink = sinks_ptr[q_head_id];
        for (int row = tid; row < valid_q_rows; row += CFG::THREADS_PER_BLOCK) {
            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, sink);
            const float exp_diff = expf(old_max - new_max);
            sRowSum[row] = exp_diff * sRowSum[row] + expf(sink - new_max);
            sRowMax[row] = new_max;
            float * sO_row = sO + row * CFG::O_STRIDE;
            for (int i = 0; i < CFG::O_STRIDE / 4; ++i) {
                float4 v = reinterpret_cast<float4 *>(sO_row)[i];
                v.x *= exp_diff;
                v.y *= exp_diff;
                v.z *= exp_diff;
                v.w *= exp_diff;
                reinterpret_cast<float4 *>(sO_row)[i] = v;
            }
        }
        __syncthreads();
    }

    // Normalize O by the running row sum and store fp32 output rows.
    if (debug_out != nullptr && debug && threadIdx.x == 0 && blockIdx.z == 1 && blockIdx.x == 0) {
        debug_out[50] = sRowMax[0];
        debug_out[51] = sRowSum[0];
        for (int c = 0; c < 8; ++c) {
            debug_out[52 + c] = sO[c];
        }
    }
    const int total_f32x4 = valid_q_rows * (D / 4);
    for (int i = tid; i < total_f32x4; i += CFG::THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i - row * (D / 4)) * 4;
        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float inv_sum = 1.0f / sum_clamped;
        const float * sO_row = sO + row * CFG::O_STRIDE;

        float4 out;
        out.x = sO_row[col + 0] * inv_sum;
        out.y = sO_row[col + 1] * inv_sum;
        out.z = sO_row[col + 2] * inv_sum;
        out.w = sO_row[col + 3] * inv_sum;
        if (debug_out != nullptr && debug && threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0 && row == 0) {
            debug_out[70] = out.x;
            debug_out[71] = out.y;
            debug_out[72] = out.z;
            debug_out[73] = out.w;
            debug_out[74] = sum_clamped;
            debug_out[75] = (float) stride_D1;
            debug_out[76] = (float) stride_D2;
        }
        *reinterpret_cast<float4 *>(
            d_base + (int64_t) (start_row + row) * stride_D2 + (int64_t) q_head_id * stride_D1 + (int64_t) col * sizeof(float)) = out;
        if (debug_out != nullptr && debug && threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0 && row == 0) {
            const float4 back = *reinterpret_cast<const float4 *>(
                d_base + (int64_t) start_row * stride_D2 + (int64_t) q_head_id * stride_D1);
            debug_out[90] = back.x;
            debug_out[91] = back.y;
            debug_out[92] = back.z;
            debug_out[93] = back.w;
            debug_out[94] = (float) (uintptr_t) (d_base - dst_ptr);
        }
    }
    __syncthreads();

#endif // FLASH_ATTN_AVAILABLE
}

// ---------------------------------------------------------------------------
// Host dispatch gate + launcher.
// ---------------------------------------------------------------------------
static bool ggml_cuda_flash_attn_ext_v100_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_V100_FA");
        return !(s && s[0] == '0'); // default ON; GGML_V100_FA=0 = kill-switch
    }();
    return v;
}

bool ggml_cuda_flash_attn_ext_v100_available(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (!ggml_cuda_flash_attn_ext_v100_enabled()) {
        return false;
    }
    if (cc != GGML_CUDA_CC_VOLTA && cc != GGML_CUDA_CC_TURING) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || Q->ne[2] < K->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[2] != 1) {
        return false;
    }
    if (K->ne[1] < Q->ne[1]) {
        return false; // causal prefill expects KV length >= query length
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false; // no ALiBi / softcap in v1
    }

    // 16-byte alignment for vectorized loads and stores.
    for (const ggml_tensor * t : {Q, K, V, mask, dst}) {
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    if (Q->nb[1] < 256 * 4 || K->nb[1] < 256 * 2 || V->nb[1] < 256 * 2) {
        return false;
    }
    return true;
}

template <typename CFG>
static void ggml_cuda_flash_attn_ext_v100_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace ggml_cuda_fattn_v100; // launch syntax needs the bare kernel name
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(ggml_cuda_flash_attn_ext_v100_available(dst));
    if (ggml_cuda_fattn_v100::v100_debug()) {
        fprintf(stderr, "[V100-MASK] ne=(%lld,%lld,%lld,%lld) nb=(%zu,%zu,%zu,%zu) K-ne1=%lld Q-ne1(M)=%lld\n",
            (long long) mask->ne[0], (long long) mask->ne[1], (long long) mask->ne[2], (long long) mask->ne[3],
            mask->nb[0], mask->nb[1], mask->nb[2], mask->nb[3],
            (long long) K->ne[1], (long long) Q->ne[1]);
    }

    const int id = ggml_cuda_get_device();
    const int H = (int) Q->ne[2];
    const int H_KV = (int) K->ne[2];
    const int M = (int) Q->ne[1];
    const int N = (int) K->ne[1];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ggml_cuda_pool_alloc<float> debug_buf(ctx.pool()); // null unless debugging
    float * debug_out = nullptr;
    if (ggml_cuda_fattn_v100::v100_debug()) {
        debug_buf.alloc(128);
        debug_out = debug_buf.ptr;
    }

    cudaStream_t main_stream = ctx.stream();

    // Per-row visibility ceiling from the mask (causal/SWA KV skip). Runs inside
    // the capture region, so the buffer is allocated from the context pool here
    // (before capture starts).
    ggml_cuda_pool_alloc<int> kv_max_buf(ctx.pool());
    int * kv_max_ptr = nullptr;
    if (mask != nullptr) {
        kv_max_buf.alloc(M);
        kv_max_ptr = kv_max_buf.ptr;
        const dim3 kv_blocks(M, 1, 1);
        const dim3 kv_threads(256, 1, 1);
        fattn_v100_mask_kvmax_kernel<<<kv_blocks, kv_threads, 0, main_stream>>>(
            (const half *) mask->data, kv_max_ptr, N, M, (int64_t) mask->nb[1]);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(CFG::THREADS_PER_BLOCK, 1, 1);
    const dim3 blocks_num((M + CFG::BLOCK_M - 1) / CFG::BLOCK_M, 1, H);

    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attn_ext_v100_kernel<CFG>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int) CFG::SMEM_BYTES));
        shared_memory_limit_raised[id] = true;
    }

    flash_attn_ext_v100_kernel<CFG><<<blocks_num, block_dim, CFG::SMEM_BYTES, main_stream>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        (const char *) mask->data,
        kv_max_ptr,
        sinks ? (const float *) sinks->data : nullptr,
        (char *) dst->data,
        M, N, H, H_KV,
        (int32_t) Q->nb[1], (int64_t) Q->nb[2], (int64_t) Q->nb[3],
        (int32_t) K->nb[1], (int64_t) K->nb[2], (int64_t) K->nb[3],
        (int32_t) V->nb[1], (int64_t) V->nb[2], (int64_t) V->nb[3],
        (int32_t) mask->nb[1], (int64_t) 0,  (int64_t) mask->nb[3],
        (int32_t) dst->nb[1], (int64_t) dst->nb[2], (int64_t) dst->nb[3],
        scale, debug_out, ggml_cuda_fattn_v100::v100_debug());
    CUDA_CHECK(cudaGetLastError());
    if (debug_out != nullptr) {
        float hbuf[128];
        CUDA_CHECK(cudaMemcpyAsync(hbuf, debug_out, sizeof(float) * 128,
            cudaMemcpyDeviceToHost, main_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
        fprintf(stderr, "[V100-DUMP] Q0: ");
        for (int i = 0; i < 8; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, " | K0: ");
        for (int i = 8; i < 16; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, " | K16: ");
        for (int i = 16; i < 24; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, "\n[V100-DUMP] sS0: ");
        for (int i = 24; i < 32; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, " | sS1: ");
        for (int i = 32; i < 40; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, "\n[V100-DUMP] P0: ");
        for (int i = 40; i < 48; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, " | max=%.3f sum=%.6f\n", hbuf[48], hbuf[49]);
        fprintf(stderr, "[V100-DUMP] final max=%.3f sum=%.6f | O0: ", hbuf[50], hbuf[51]);
        for (int i = 52; i < 60; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, "\n[V100-DUMP] kernel wrote out: ");
        for (int i = 70; i < 74; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, "| sum=%.3f sD1=%.0f sD2=%.0f\n", hbuf[74], hbuf[75], hbuf[76]);
        float drb[8] = {0};
        CUDA_CHECK(cudaMemcpyAsync(drb, dst->data, sizeof(float) * 8,
            cudaMemcpyDeviceToHost, main_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
        fprintf(stderr, "[V100-DUMP] dst[0..7] readback: ");
        for (int i = 0; i < 8; ++i) fprintf(stderr, "%.3f ", drb[i]);
        fprintf(stderr, "\n[V100-DUMP] kernel readback same addr: ");
        for (int i = 90; i < 94; ++i) fprintf(stderr, "%.3f ", hbuf[i]);
        fprintf(stderr, "| d_base offset=%.0f\n", hbuf[94]);
        fprintf(stderr, "[V100-DUMP] visible-cols r0=%d r1=%d sS00=%.2f sS0_16=%.2f N=%.0f M=%.0f valid_q_rows=%.0f\n",
            (int) hbuf[0], (int) hbuf[1], hbuf[2], hbuf[3], hbuf[4], hbuf[5], hbuf[6], hbuf[7]);
        fprintf(stderr, "[V100-DUMP] QK-maskhit gm=%.0f gn=%.0f acc=%.2f mask[0][0..3]: ",
            hbuf[20], hbuf[21], hbuf[22]);
        for (int i = 30; i < 34; ++i) fprintf(stderr, "%.2f ", hbuf[i]);
        fprintf(stderr, "| direct mask[0][0..3]: ");
        for (int i = 96; i < 100; ++i) fprintf(stderr, "%.2f ", hbuf[i]);
        fprintf(stderr, "\n");
    }
}

static void ggml_cuda_flash_attn_ext_v100(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc == GGML_CUDA_CC_TURING) {
        using namespace ggml_cuda_fattn_v100;
        ggml_cuda_flash_attn_ext_v100_launch<cfg_sm75>(ctx, dst);
    } else {
        using namespace ggml_cuda_fattn_v100;
        ggml_cuda_flash_attn_ext_v100_launch<cfg_v100>(ctx, dst);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)