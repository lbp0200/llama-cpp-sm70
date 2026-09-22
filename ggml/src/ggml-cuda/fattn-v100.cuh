#pragma once
// Volta (SM70) flash attention: 1Cat FLASH_ATTN_V100 dataflow ported to ggml.
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
// - CUDA only (no HIP/MUSA). cc == Volta (700) only.
// - D = 256, fp16 K/V, fp32 Q input, fp32 output (llama.cpp FA conventions).
// - Causal / SWA via the f16 mask tensor (added after scale, like the VEC path).
// - Single sequence (ne[3] == 1). GQA handled in-kernel via gqa_ratio.
// - Env kill-switch: GGML_V100_FA=0 falls back to the existing kernels.

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <mma.h>

using namespace nvcuda::wmma;

namespace ggml_cuda_fattn_v100 {

// WMMA tile shape used by 1Cat (validated against nvcuda::wmma by their harness).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BLOCK_M = 32;
static constexpr int BLOCK_N = 64;
static constexpr int WARPS_PER_BLOCK = 16;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
static constexpr int THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M;
static constexpr int P_SUB_TILE = 32;
static constexpr int D256_PAD = 0; // (8 - (256 % 32) + 32) % 32

static constexpr int Q_STRIDE  = 256 + D256_PAD;
static constexpr int KV_STRIDE = 256 + D256_PAD;
static constexpr int S_STRIDE  = BLOCK_N + D256_PAD;
static constexpr int P_STRIDE  = P_SUB_TILE + D256_PAD;
static constexpr int O_STRIDE  = 256 + D256_PAD;

static constexpr float NEG_INF = -1e30f;
static constexpr float RCP_LN2 = 1.4426950408889634f; // unused for D=256, kept for clarity

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

__device__ __forceinline__ void init_smem_v100(char * smem_raw) {
    constexpr int N_U4 = SMEM_BYTES / 16;
    const int tid = threadIdx.x;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
#pragma unroll 1
    for (int i = tid; i < N_U4; i += THREADS_PER_BLOCK) {
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(addr + (i << 4)),
                     "r"(0)
                     : "memory");
    }
    __syncthreads();
}

// Convert fp32 Q rows (llama.cpp graph convention) to fp16 smem tiles.
__device__ __forceinline__ void load_q_f32_to_smem(
        const char * __restrict__ q_ptr, const int q_row_stride,
        half * __restrict__ sQ, const int valid_q_rows) {
        constexpr int D_H2 = 256 / 2; // half2 per row after conversion
    const int tid = threadIdx.x;
    const int total = valid_q_rows * D_H2;
    const int row_stride_u2 = q_row_stride / (2 * (int) sizeof(float));

    const float2 * q_vec = reinterpret_cast<const float2 *>(q_ptr);
    half2 * sQ_h2 = reinterpret_cast<half2 *>(sQ);

    for (int idx = tid; idx < total; idx += THREADS_PER_BLOCK) {
        const int row = idx / D_H2;
        const int vec = idx - row * D_H2;
        if (row < valid_q_rows) {
            const float2 f2 = __ldg(&q_vec[row * row_stride_u2 + vec]);
            sQ_h2[row * (Q_STRIDE / 2) + vec] = __floats2half2_rn(f2.x, f2.y);
        }
    }
}

// f16 K/V tiles from gmem into smem, uint4 vectorized __ldg loads.
__device__ __forceinline__ void load_kv_f16_to_smem(
        const char * __restrict__ kv_ptr, const int64_t kv_row_stride,
        half * __restrict__ sKV, const int valid_k_rows) {
    constexpr int PER_UINT4 = 8; // halfs per 16-byte load
    const int tid = threadIdx.x;
    const int d_u4 = 256 / PER_UINT4;
    const int row_stride_u4 = (int) (kv_row_stride / (PER_UINT4 * (int) sizeof(half)));

    const uint4 * kv_vec = reinterpret_cast<const uint4 *>(kv_ptr);
    uint4 * sKV_vec = reinterpret_cast<uint4 *>(sKV);

    for (int idx = tid; idx < (valid_k_rows * d_u4); idx += THREADS_PER_BLOCK) {
        const int row = idx / d_u4;
        const int vec = idx - row * d_u4;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < valid_k_rows) {
            val = __ldg(&kv_vec[row * row_stride_u4 + vec]);
        }
        sKV_vec[row * (KV_STRIDE / PER_UINT4) + vec] = val;
    }
}

} // namespace ggml_cuda_fattn_v100

// ---------------------------------------------------------------------------
// Device kernel. One block handles BLOCK_M query rows of one (batch, head) and
// walks the full causal KV range in BLOCK_N chunks.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(ggml_cuda_fattn_v100::THREADS_PER_BLOCK, 2)
flash_attn_ext_v100_kernel(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const float * __restrict__ sinks_ptr,
        char       * __restrict__ dst_ptr,
        const int32_t M, const int32_t N,
        const int32_t H, const int32_t H_KV,
        const int32_t stride_Q1, const int64_t stride_Q2, const int64_t stride_Q3,
        const int32_t stride_K1, const int64_t stride_K2, const int64_t stride_K3,
        const int32_t stride_V1, const int64_t stride_V2, const int64_t stride_V3,
        const int32_t stride_M1, const int64_t stride_M2, const int64_t stride_M3,
        const int32_t stride_D1, const int64_t stride_D2, const int64_t stride_D3,
        const float scale) {
    using namespace ggml_cuda_fattn_v100;
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D = 256;
    const int tid = threadIdx.x;

    const int batch_head_id = blockIdx.z;
    if (batch_head_id >= H) {
        return;
    }
    const int batch_id = batch_head_id / H;
    const int q_head_id = batch_head_id - batch_id * H;
    const int kv_group_size = H / H_KV;
    const int kv_head_id = q_head_id / kv_group_size;

    const int start_row = blockIdx.x * BLOCK_M;
    if (start_row >= M) {
        return;
    }

    const int valid_q_rows = min(BLOCK_M, M - start_row);

    // v1 correctness: walk the full KV range and let the mask decide visibility.
    // Geometric causal truncation is NOT safe here: masks can be all-visible or
    // sink-shaped, and truncating would drop valid keys. (A causal-only fast path
    // that scans the mask for the per-row CEIL is a later perf optimization.)
    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    const char * q_base = Q_ptr + stride_Q3 * batch_id + stride_Q2 * q_head_id;
    const char * k_base = K_ptr + stride_K3 * batch_id + stride_K2 * kv_head_id;
    const char * v_base = V_ptr + stride_V3 * batch_id + stride_V2 * kv_head_id;
    const char * m_base = mask_ptr + stride_M3 * batch_id + stride_M2 * 0;
    // FA output is permute(0,2,1,3): bytes map as [D][H][M][B], so the head
    // stride is nb[1] and the token stride is nb[2] (see ggml_flash_attn_ext).
    char *       d_base = dst_ptr + stride_D3 * batch_id;

    extern __shared__ char smem_raw[];
    init_smem_v100(smem_raw);
    SmemLayout & smem = *reinterpret_cast<SmemLayout *>(smem_raw);

    half  * sQ = smem.q;
    half  * sK = smem.kv.k;
    half  * sV = smem.kv.v;
    float * sS = smem.s;
    half  * sP = smem.p;
    float * sO = smem.o;
    float * sRowMax = smem.row_max;
    float * sRowSum = smem.row_sum;

    if (threadIdx.x < BLOCK_M) {
        sRowMax[threadIdx.x] = NEG_INF;
    }

    load_q_f32_to_smem(q_base + (int64_t) start_row * stride_Q1, stride_Q1, sQ, valid_q_rows);
    __syncthreads();

    for (int block_n = 0; block_n < num_n_tiles; ++block_n) {
        const int start_col = block_n * BLOCK_N;
        const int valid_k_rows = min(BLOCK_N, N - start_col);

        // QK: load K tile, 16 warps x m16n16k16 over D, store fp32 scores to smem.
        load_kv_f16_to_smem(k_base + (int64_t) start_col * stride_K1, stride_K1, sK, valid_k_rows);
        __syncthreads();

        const int num_tiles_m_qk = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_n_qk = (BLOCK_N + WMMA_N - 1) / WMMA_N;
        const int num_tiles_k_qk = (D + WMMA_K - 1) / WMMA_K;
        const int total_tiles_qk = num_tiles_m_qk * num_tiles_n_qk;
        const int tiles_per_warp_qk = (total_tiles_qk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        // nvcuda::wmma m16n16k16 fragment -> (row, col) permute, from 1Cat.
        const unsigned row_causal = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
        const unsigned col_causal = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;

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
                load_matrix_sync(a_frag, sQ + tile_m * Q_STRIDE + k_off, Q_STRIDE);
                load_matrix_sync(b_frag, sK + tile_n * KV_STRIDE + k_off, KV_STRIDE);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

#pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                const unsigned col = col_causal + (i & 0b1) + ((i >> 2) & 0b1) * 4;
                const unsigned row = row_causal + ((i >> 1) & 0b1) * 2;

                const int global_m = start_row + tile_m + row;
                const int global_n = start_col + tile_n + col;
                const bool is_valid = (global_m < start_row + valid_q_rows) &&
                                      (global_n < start_col + valid_k_rows);

                if (is_valid) {
                    const half mask_val = __ldg(reinterpret_cast<const half *>(m_base + (int64_t) global_m * stride_M1) + global_n);
                    acc_frag.x[i] = acc_frag.x[i] * scale + __half2float(mask_val);
                } else {
                    acc_frag.x[i] = NEG_INF;
                }
            }
            store_matrix_sync(sS + tile_m * S_STRIDE + tile_n, acc_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();
        // PV: load V into the K/V union slot, sub-tile softmax, WMMA PV.
        load_kv_f16_to_smem(v_base + (int64_t) start_col * stride_V1, stride_V1, sV, valid_k_rows);
        __syncthreads();

        const int num_tiles_m_pv = (BLOCK_M + WMMA_M - 1) / WMMA_M;
        const int num_tiles_d_pv = (D + WMMA_N - 1) / WMMA_N;
        const int total_tiles_pv = num_tiles_m_pv * num_tiles_d_pv;
        const int tiles_per_warp_pv = (total_tiles_pv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int p_tile_capacity = P_SUB_TILE;

        for (int sub_start = 0; sub_start < valid_k_rows; sub_start += p_tile_capacity) {
            const int sub_valid_k_rows = min(p_tile_capacity, valid_k_rows - sub_start);

            if (tid < valid_q_rows * THREADS_PER_ROW) {
                const int row = tid / THREADS_PER_ROW;
                const int thread_in_row = tid - row * THREADS_PER_ROW;
                const unsigned sync_mask =
                    (valid_q_rows == BLOCK_M) ? 0xFFFFFFFFU : __activemask();
                const int row_leader = __ffs(sync_mask) - 1;

                float * sS_row_f = sS + row * S_STRIDE + sub_start;
                half  * sP_row_h = sP + row * P_STRIDE;

                const int vec_cols = sub_valid_k_rows >> 2;
                const int vecs_per_thread = (vec_cols + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
                const int tail_start = vec_cols << 2;

                float thread_max = NEG_INF;
                float4 * sS_vec4 = reinterpret_cast<float4 *>(sS_row_f);

#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j) {
                    const int vc = thread_in_row + j * THREADS_PER_ROW;
                    if (vc < vec_cols) {
                        const float4 v4 = sS_vec4[vc];
                        thread_max = fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
                    }
                }
#pragma unroll 4
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows; c += THREADS_PER_ROW) {
                    thread_max = fmaxf(thread_max, sS_row_f[c]);
                }
#pragma unroll
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_max = fmaxf(thread_max, __shfl_down_sync(sync_mask, thread_max, o, THREADS_PER_ROW));
                }

                const float row_max = __shfl_sync(sync_mask, thread_max, row_leader, THREADS_PER_ROW);
                const float old_max = sRowMax[row];
                const float new_max = fmaxf(old_max, row_max);
                const float exp_diff = expf(fmaxf(old_max - new_max, -80.0f));

                float thread_sum = 0.0f;
                half2 half_buffer[20];
                int vc_base = thread_in_row;
                int h2_idx = 0;
                int tail_col = -1;
                half tail_value = __float2half(0.f);

#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
                    if (vc_base < vec_cols) {
                        const float4 v4 = sS_vec4[vc_base];
                        const float e0 = expf(fmaxf(v4.x - new_max, -80.0f));
                        const float e1 = expf(fmaxf(v4.y - new_max, -80.0f));
                        const float e2 = expf(fmaxf(v4.z - new_max, -80.0f));
                        const float e3 = expf(fmaxf(v4.w - new_max, -80.0f));
                        thread_sum += (e0 + e1) + (e2 + e3);
                        half_buffer[h2_idx++] = __float22half2_rn(make_float2(e0, e1));
                        half_buffer[h2_idx++] = __float22half2_rn(make_float2(e2, e3));
                    }
                }
#pragma unroll 4
                for (int c = tail_start + thread_in_row; c < sub_valid_k_rows; c += THREADS_PER_ROW) {
                    const float e = expf(fmaxf(sS_row_f[c] - new_max, -80.0f));
                    thread_sum += e;
                    tail_col = c;
                    tail_value = __float2half_rn(e);
                }
#pragma unroll
                for (int o = THREADS_PER_ROW / 2; o > 0; o >>= 1) {
                    thread_sum += __shfl_down_sync(sync_mask, thread_sum, o, THREADS_PER_ROW);
                }

                const float row_sum = __shfl_sync(sync_mask, thread_sum, row_leader, THREADS_PER_ROW);
                if (thread_in_row == 0) {
                    sRowSum[row] = exp_diff * sRowSum[row] + row_sum;
                    sRowMax[row] = new_max;
                }

                h2_idx = 0;
                vc_base = thread_in_row;
                half2 * sP_half2 = reinterpret_cast<half2 *>(sP_row_h);
#pragma unroll 4
                for (int j = 0; j < vecs_per_thread; ++j, vc_base += THREADS_PER_ROW) {
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
                for (int c = tail_start + thread_in_row; c < p_tile_capacity; c += THREADS_PER_ROW) {
                    if (c >= sub_valid_k_rows) {
                        sP_row_h[c] = __float2half(0.f);
                    }
                }

                if (block_n > 0 || sub_start > 0) {
                    float * sO_row = sO + row * O_STRIDE;
                    float4 * sO_vec = reinterpret_cast<float4 *>(sO_row);
                    const int o_vec_count = (O_STRIDE + 3) >> 2;
#pragma unroll 4
                    for (int ov = thread_in_row; ov < o_vec_count; ov += THREADS_PER_ROW) {
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

            // PV WMMA: a = p (fp16, row-major), b = V (fp16, row-major), acc in fp32.
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

                load_matrix_sync(acc_frag, sO + tile_m * O_STRIDE + tile_d, O_STRIDE, mem_row_major);

#pragma unroll
                for (int tile_k = 0; tile_k < num_tiles_k_pv; ++tile_k) {
                    const int k_off = tile_k * WMMA_K;
                    if (k_off >= sub_valid_k_rows) {
                        break;
                    }
                    load_matrix_sync(a_frag, sP + tile_m * P_STRIDE + k_off, P_STRIDE);
                    load_matrix_sync(b_frag, sV + (sub_start + k_off) * KV_STRIDE + tile_d, KV_STRIDE);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
                store_matrix_sync(sO + tile_m * O_STRIDE + tile_d, acc_frag, O_STRIDE, mem_row_major);
            }
            __syncthreads();
        }
    }

    // Sinks: an extra no-V softmax term per head. Same math as the VEC kernels:
    // max/sum absorb exp(sink), and the accumulated O is rescaled by exp_diff.
    if (sinks_ptr != nullptr) {
        const float sink = sinks_ptr[q_head_id];
        for (int row = tid; row < valid_q_rows; row += THREADS_PER_BLOCK) {
            const float old_max = sRowMax[row];
            const float new_max = fmaxf(old_max, sink);
            const float exp_diff = expf(old_max - new_max);
            sRowSum[row] = exp_diff * sRowSum[row] + expf(sink - new_max);
            sRowMax[row] = new_max;
            float * sO_row = sO + row * O_STRIDE;
            for (int i = 0; i < O_STRIDE / 4; ++i) {
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
    const int total_f32x4 = valid_q_rows * (D / 4);
    for (int i = tid; i < total_f32x4; i += THREADS_PER_BLOCK) {
        const int row = i / (D / 4);
        const int col = (i - row * (D / 4)) * 4;
        const float sum_clamped = fmaxf(sRowSum[row], 1e-24f);
        const float inv_sum = 1.0f / sum_clamped;
        const float * sO_row = sO + row * O_STRIDE;

        float4 out;
        out.x = sO_row[col + 0] * inv_sum;
        out.y = sO_row[col + 1] * inv_sum;
        out.z = sO_row[col + 2] * inv_sum;
        out.w = sO_row[col + 3] * inv_sum;
        *reinterpret_cast<float4 *>(
            d_base + (int64_t) (start_row + row) * stride_D2 + (int64_t) q_head_id * stride_D1 + (int64_t) col * sizeof(float)) = out;
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
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || Q->ne[2] % K->ne[2] != 0) {
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

void ggml_cuda_flash_attn_ext_v100(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(ggml_cuda_flash_attn_ext_v100_available(dst));

    const int id = ggml_cuda_get_device();
    const int H = (int) Q->ne[2];
    const int H_KV = (int) K->ne[2];
    const int M = (int) Q->ne[1];
    const int N = (int) K->ne[1];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t main_stream = ctx.stream();

    const dim3 block_dim(ggml_cuda_fattn_v100::THREADS_PER_BLOCK, 1, 1);
    const dim3 blocks_num((M + ggml_cuda_fattn_v100::BLOCK_M - 1) / ggml_cuda_fattn_v100::BLOCK_M, 1, H);

    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attn_ext_v100_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ggml_cuda_fattn_v100::SMEM_BYTES));
        shared_memory_limit_raised[id] = true;
    }

    flash_attn_ext_v100_kernel<<<blocks_num, block_dim, ggml_cuda_fattn_v100::SMEM_BYTES, main_stream>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        (const char *) mask->data,
        sinks ? (const float *) sinks->data : nullptr,
        (char *) dst->data,
        M, N, H, H_KV,
        (int32_t) Q->nb[1], (int64_t) Q->nb[2], (int64_t) Q->nb[3],
        (int32_t) K->nb[1], (int64_t) K->nb[2], (int64_t) K->nb[3],
        (int32_t) V->nb[1], (int64_t) V->nb[2], (int64_t) V->nb[3],
        (int32_t) mask->nb[1], (int64_t) 0,  (int64_t) mask->nb[3],
        (int32_t) dst->nb[1], (int64_t) dst->nb[2], (int64_t) dst->nb[3],
        scale);
    CUDA_CHECK(cudaGetLastError());
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)