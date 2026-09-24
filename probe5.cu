// probe5.cu : validate the mma-atom operand directions for the route-B engine
// on sm_75 before porting any kernel code. One warp.
//
// QK: score(i,j) = sum_k Q[i][k]*K[j][k]; D tile<16,16,float> via
// mma(tile<16,16,float>&, tile<16,8,half2>&, tile<16,8,half2>&).
// Hypotheses for the operand loads:
//   H1: A = plain  ldmatrix from Q  [qrow][k]  ; B = plain  ldmatrix from Kt [k][kv]
//   H2: A and B args swapped in mma() with the same loads
//   H3: A = plain  ldmatrix from Q  [qrow][k]  ; B = ldmatrix_trans from K [kv][k]
// Each is expanded with get_i/get_j both ways (i=Q vs i=KV); the correct
// direction shows err ~ 0 in one of the two mappings.
//
// PV: O(i,d) = sum_kv P[i][kv]*V[kv][d] with P = get_half2(D from H1).
//   PV1: B = plain  ldmatrix from V [kv][d16]
//   PV2: B = ldmatrix_trans from V [kv][d16]
//   each tested with mma(D2, P, V) and mma(D2, V, P) args.
//
// Compile: nvcc -arch=sm_75 -DGGML_USE_CUDA -I ggml/include -I ggml/src probe5.cu -o probe5 && ./probe5
#include <cstdio>
#include <cstdarg>
#include <cuda_runtime.h>

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mma.cuh"

// Stubs for ggml-cuda.cu symbols pulled in by common.cuh (not used by the probe).
void ggml_cuda_error(const char * ctx, const char * info, const char * file, int line, const char * func) {
    fprintf(stderr, "[ggml_cuda_error] %s %s %s:%d %s\n", ctx ? ctx : "", info ? info : "", file, line, func ? func : "");
    abort();
}

int ggml_cuda_get_device() {
    return 0;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "[ggml_abort] %s:%d ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

using namespace ggml_cuda_mma;

static constexpr int QS  = 132; // h2 stride of Q/K/V rows (128 h2 + 4 pad)
static constexpr int KTS = 20;  // h2 stride of the K^T buffer (16 kv + 4 pad)

__global__ void probe5(float * out) {
    __shared__ half2 sQ[16 * QS];
    __shared__ half2 sK[16 * QS];
    __shared__ half2 sKt[128 * KTS];
    __shared__ half2 sV[16 * QS];
    __shared__ half2 sPsm[16 * 12]; // P: [Qrow][kv-pair], stride 12 (8+4 pad)
    __shared__ half2 sVT[16 * 12];  // V^T: [d][kv-pair], stride 12

    const int lane = threadIdx.x;

    // Deterministic small-integer data: products stay exact in f16.
    // Non-degenerate integer patterns: rows i*131 mod 23 are distinct for all
    // i < 23, so every row differs (the old j*5 mod 5 pattern collapsed all K rows).
    for (int idx = lane; idx < 16 * QS; idx += 32) {
        const int i = idx / QS;
        const int e = idx % QS;
        const int k = 2 * e;
        const int vq = ((i * 131 +      k * 17) % 23) - 11;
        const int vk = ((i * 137 + (k + 1) * 19) % 23) - 11;
        const int vv = ((i * 139 + (k + 3) * 29) % 23) - 11;
        sQ[idx] = make_half2(__float2half((float) vq), __float2half((float) (((i * 131 + (k + 1) * 17) % 23) - 11)));
        sK[idx] = make_half2(__float2half((float) vk), __float2half((float) (((i * 137 + (k + 2) * 19) % 23) - 11)));
        sV[idx] = make_half2(__float2half((float) vv), __float2half((float) (((i * 139 + (k + 4) * 29) % 23) - 11)));
    }
    // K^T: rows = k pairs (128), cols = 16 KV, stride KTS.
    for (int idx = lane; idx < 128 * KTS; idx += 32) {
        const int kh = idx / KTS;   // k pair index
        const int j  = idx % KTS;   // kv index (>=16 is padding)
        if (j < 16) {
            const int k = 2 * kh;
            sKt[idx] = make_half2(__float2half((float) (((j * 137 + (k + 1) * 19) % 23) - 11)),
                                  __float2half((float) (((j * 137 + (k + 2) * 19) % 23) - 11)));
        } else {
            sKt[idx] = make_half2(0.f, 0.f);
        }
    }
    __syncthreads();

    // Host-independent expected scores and O, computed by plain loops.
    __shared__ float exp_s[16 * 16];
    __shared__ float exp_o[16 * 16];
    for (int idx = lane; idx < 16 * 16; idx += 32) {
        const int i = idx / 16;
        const int j = idx % 16;
        auto h2f = [](const half2 & v, int e) { return __half2float(e ? v.y : v.x); };
        float acc = 0.f;
        for (int k = 0; k < 256; ++k) {
            acc += h2f(sQ[i * QS + k / 2], k & 1) * h2f(sK[j * QS + k / 2], k & 1);
        }
        exp_s[idx] = acc;
        float o = 0.f;
        // O(i,d) = sum_kv f16(score(i,kv)) * V[kv][d]; d = j
        for (int kv = 0; kv < 16; ++kv) {
            const float p_h = __half2float(__float2half(exp_s[i * 16 + kv]));
            o += p_h * h2f(sV[kv * QS + j / 2], j & 1);
        }
        exp_o[idx] = o;
    }
    __syncthreads();

    // H6 operands: P round-tripped through smem as [Qrow][kv-pair] (the exact
    // proven QK-A geometry) and V transposed to [d][kv-pair] (the QK-B geometry).
    for (int idx = lane; idx < 16 * 8; idx += 32) {
        const int i = idx / 8;
        const int t = idx % 8;
        float a0 = 0.f, a1 = 0.f;
        for (int k = 0; k < 256; ++k) {
            const half qh = sQ[i * QS + k / 2].x; // placeholder, replaced below
            (void) qh;
            break;
        }
        // full dot for kv = 2t and 2t+1
        for (int k = 0; k < 256; ++k) {
            const half2 q = sQ[i * QS + k / 2];
            const half2 k0 = sK[(2 * t) * QS + k / 2];
            const half2 k1 = sK[(2 * t + 1) * QS + k / 2];
            const float qv0 = (k & 1) ? __half2float(q.y) : __half2float(q.x);
            a0 += qv0 * ((k & 1) ? __half2float(k0.y) : __half2float(k0.x));
            a1 += qv0 * ((k & 1) ? __half2float(k1.y) : __half2float(k1.x));
        }
        sPsm[idx] = make_half2(__float2half(a0), __float2half(a1));
    }
    for (int idx = lane; idx < 16 * 8; idx += 32) {
        const int d = idx / 8;   // d half index 0..15
        const int t = idx % 8;   // kv pair
        const half2 v0 = sV[(2 * t) * QS + d / 2];
        const half2 v1 = sV[(2 * t + 1) * QS + d / 2];
        const half e0 = (d & 1) ? v0.y : v0.x;
        const half e1 = (d & 1) ? v1.y : v1.x;
        sVT[idx] = make_half2(e0, e1);
    }
    __syncthreads();

    auto ldm_plain = [](half2 * base, int stride) {
        tile<16, 8, half2> t;
        load_ldmatrix(t, base, stride);
        return t;
    };
    auto ldm_trans = [](half2 * base, int stride) {
        tile<16, 8, half2> t;
        load_ldmatrix_trans(t, base, stride);
        return t;
    };

    // Accumulate max abs error per hypothesis/mapping into out[0..9].
    __shared__ float errs[10];
    __shared__ float rawd[8];
    if (lane == 0) {
        for (int e = 0; e < 10; ++e) errs[e] = 0.f;
    }
    __syncthreads();

    // ---- QK hypotheses ----
    {
        tile<16, 16, float> D;
        for (int l = 0; l < D.ne; ++l) D.x[l] = 0.f;
        for (int k0h = 0; k0h < 128; k0h += 8) { // full D=256 halves, 16 per mma
            tile<16, 8, half2> A = ldm_plain(&sQ[0] + k0h, QS);
            tile<16, 8, half2> B = ldm_plain(&sK[0] + k0h, QS);
            mma(D, A, B);
        }
        if (lane == 0) {
            for (int r = 0; r < 8; ++r) {
                rawd[r] = D.x[r];
            }
        }
        for (int l = 0; l < D.ne; ++l) {
            const float eA = fabsf(D.x[l] - exp_s[D.get_i(l) * 16 + D.get_j(l)]);
            const float eB = fabsf(D.x[l] - exp_s[D.get_j(l) * 16 + D.get_i(l)]);
            atomicMax((int *) &errs[0], __float_as_int(eA));
            atomicMax((int *) &errs[1], __float_as_int(eB));
        }
    }
    {
        tile<16, 16, float> D;
        for (int l = 0; l < D.ne; ++l) D.x[l] = 0.f;
        for (int k0h = 0; k0h < 128; k0h += 8) {
            tile<16, 8, half2> A = ldm_plain(&sQ[0] + k0h, QS);
            tile<16, 8, half2> B = ldm_plain(&sK[0] + k0h, QS);
            mma(D, B, A); // swapped arg order
        }
        for (int l = 0; l < D.ne; ++l) {
            const float eA = fabsf(D.x[l] - exp_s[D.get_i(l) * 16 + D.get_j(l)]);
            const float eB = fabsf(D.x[l] - exp_s[D.get_j(l) * 16 + D.get_i(l)]);
            atomicMax((int *) &errs[2], __float_as_int(eA));
            atomicMax((int *) &errs[3], __float_as_int(eB));
        }
    }
    {
        tile<16, 16, float> D;
        for (int l = 0; l < D.ne; ++l) D.x[l] = 0.f;
        for (int k0h = 0; k0h < 128; k0h += 8) {
            tile<16, 8, half2> A = ldm_plain(&sQ[0] + k0h, QS);
            tile<16, 8, half2> B = ldm_trans(&sK[0] + k0h, QS);
            mma(D, A, B);
        }
        for (int l = 0; l < D.ne; ++l) {
            const float eA = fabsf(D.x[l] - exp_s[D.get_i(l) * 16 + D.get_j(l)]);
            const float eB = fabsf(D.x[l] - exp_s[D.get_j(l) * 16 + D.get_i(l)]);
            atomicMax((int *) &errs[4], __float_as_int(eA));
            atomicMax((int *) &errs[5], __float_as_int(eB));
        }
    }

    // ---- PV hypotheses: P = get_half2(score tile from H1) ----
    {
        tile<16, 16, float> S;
        for (int l = 0; l < S.ne; ++l) S.x[l] = 0.f;
        for (int k0h = 0; k0h < 128; k0h += 8) {
            tile<16, 8, half2> A = ldm_plain(&sQ[0] + k0h, QS);
            tile<16, 8, half2> B = ldm_plain(&sK[0] + k0h, QS);
            mma(S, A, B);
        }
        tile<16, 8, half2> P = get_half2(S);
        // R3: A = get_half2(S), B = plain load from gmem-layout sV.
        tile<16, 16, float> O;
        for (int l = 0; l < O.ne; ++l) O.x[l] = 0.f;
        {
            tile<16, 8, half2> A = P;
            tile<16, 8, half2> B = ldm_plain(&sV[0], QS);
            if (lane == 0) {
                rawd[0] = __half2float(A.x[0].x);
                rawd[1] = __half2float(A.x[0].y);
                rawd[2] = __half2float(A.x[1].x);
                rawd[3] = __half2float(A.x[1].y);
                rawd[4] = __half2float(B.x[0].x);
                rawd[5] = __half2float(B.x[0].y);
                rawd[6] = __half2float(B.x[1].x);
                rawd[7] = __half2float(B.x[1].y);
            }
            mma(O, A, B);
        }
        for (int l = 0; l < O.ne; ++l) {
            const float eA = fabsf(O.x[l] - exp_o[O.get_i(l) * 16 + O.get_j(l)]);
            const float eB = fabsf(O.x[l] - exp_o[O.get_j(l) * 16 + O.get_i(l)]);
            atomicMax((int *) &errs[6], __float_as_int(eA));
            atomicMax((int *) &errs[7], __float_as_int(eB));
        }
        // R2: upstream roles: A = ldmatrix_trans from sV, B = get_half2(S).
        tile<16, 16, float> O2;
        for (int l = 0; l < O2.ne; ++l) O2.x[l] = 0.f;
        {
            tile<16, 8, half2> A = ldm_trans(&sV[0], QS);
            tile<16, 8, half2> B = P;
            mma(O2, A, B);
        }
        for (int l = 0; l < O2.ne; ++l) {
            const float eA = fabsf(O2.x[l] - exp_o[O2.get_i(l) * 16 + O2.get_j(l)]);
            const float eB = fabsf(O2.x[l] - exp_o[O2.get_j(l) * 16 + O2.get_i(l)]);
            atomicMax((int *) &errs[8], __float_as_int(eA));
            atomicMax((int *) &errs[9], __float_as_int(eB));
        }
    }

    if (lane == 0) {
        for (int e = 0; e < 10; ++e) {
            out[e] = errs[e]; // errs holds the float bit pattern written by atomicMax
        }
        for (int r = 0; r < 8; ++r) {
            out[16 + r] = rawd[r];        // out[16..23] = H1 D.x[0..7]
        }
        out[24] = exp_s[0 * 16 + 0];
        out[25] = exp_s[0 * 16 + 1];
        out[26] = exp_s[8 * 16 + 0];
        out[27] = exp_s[0 * 16 + 8];
        out[24] = __half2float(sPsm[0].x);
        out[25] = __half2float(sPsm[0].y);
        out[26] = __half2float(sVT[0].x);
        out[27] = __half2float(sVT[0].y);
        out[28] = __half2float(sVT[1].x);
    }
}

int main() {
    float * d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float) * 32);
    probe5<<<1, 32>>>(d_out);
    cudaError_t launch_err = cudaGetLastError();
    printf("launch: %s\n", cudaGetErrorString(launch_err));
    float out[32] = {0};
    cudaError_t copy_err = cudaMemcpy(out, d_out, sizeof(float) * 32, cudaMemcpyDeviceToHost);
    printf("run/copy: %s\n", cudaGetErrorString(copy_err));
    const char * names[12] = {
        "QK H1 i=Q,j=KV", "QK H1 i=KV,j=Q",
        "QK H2 i=Q,j=KV", "QK H2 i=KV,j=Q",
        "QK H3 i=Q,j=KV", "QK H3 i=KV,j=Q",
        "R3 P,Vplain i=Q,d", "R3 i=d,Q",
        "R2 Vtrans,P i=Q,d", "R2 i=d,Q",
        "marker0", "marker1",
    };
    for (int i = 0; i < 10; ++i) {
        printf("%-20s maxerr=%.4f %s\n", names[i], out[i], out[i] < 500.0f ? "PASS" : "-");
    }
    printf("R3 lane0 Aregs(x0,y0,x1,y1) Bregs:");
    for (int i = 16; i <= 23; ++i) printf(" %.2f", out[i]);
    printf("\nsrc sPsm0x,sPsm0y,sVT0x,sVT0y,sVT1x:");
    for (int i = 24; i <= 28; ++i) printf(" %.2f", out[i]);
    printf("\n");
    return 0;
}
