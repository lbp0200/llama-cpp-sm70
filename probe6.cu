// probe6.cu : validate the cross-warp row protocol of the mma engine in
// isolation (QK/PV atoms are already proven by probe5). One block of
// (32 lanes x 8 warps) mirrors the engine topology: warps split into
// m-tiles x n16 subs (2 x 4). Synthetic per-(m,sub,row,col) scores, three
// fake n-tiles with online softmax; dumps sRowMax/sRowSum vs host reference.
// Compile: nvcc -arch=sm_75 probe6.cu -o probe6 && ./probe6
#include <cstdio>
#include <cuda_runtime.h>

static constexpr int N_SUBS = 4;
static constexpr int WARP_M = 4;
static constexpr int TILES = 3;

__host__ __device__ __forceinline__ float synth(int act, int col) {
    return (float) (((act * 7 + col * 3) % 19) - 9) * 1.5f;
}

__global__ void probe6(float * out) {
    __shared__ float scr_max[32 * N_SUBS];
    __shared__ float scr_sum[32 * N_SUBS];
    __shared__ float row_max[32];
    __shared__ float row_sum[32];

    const int lane = threadIdx.x;
    const int w_id = threadIdx.y;
    const int m_tile = w_id / WARP_M;
    const int sub = w_id % WARP_M;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 32 * N_SUBS; ++i) { scr_max[i] = -1e30f; scr_sum[i] = 0.f; }
        for (int i = 0; i < 32; ++i) { row_max[i] = -1e30f; row_sum[i] = 0.f; }
    }
    __syncthreads();

    // fi/fj with LANE id (mma.cuh tile<16,16,float> formulas)
    auto fi = [&](int l) { return ((l / 2) % 2) * 8 + lane / 4; };
    auto fj = [&](int l) { return (l / 4) * 8 + (lane % 4) * 2 + (l % 2); };

    for (int t = 0; t < TILES; ++t) {
        // synthetic masked scores for this sub's 16 cols, 8 frag slots
        float S[8];
        for (int l = 0; l < 8; ++l) {
            const int act = m_tile * 16 + fi(l);
            const int col = t * 64 + sub * 16 + fj(l);
            S[l] = synth(act, col);
        }

        // in-warp row max: rows lane/4 and 8+lane/4 over this sub's cols
        float mx_a = -1e30f, mx_b = -1e30f;
        mx_a = fmaxf(mx_a, S[0]); mx_a = fmaxf(mx_a, S[1]);
        mx_a = fmaxf(mx_a, S[4]); mx_a = fmaxf(mx_a, S[5]);
        mx_b = fmaxf(mx_b, S[2]); mx_b = fmaxf(mx_b, S[3]);
        mx_b = fmaxf(mx_b, S[6]); mx_b = fmaxf(mx_b, S[7]);
        // combine via xor shuffles: plain assignment would leave lane0 with
        // only lane3's value (root cause of the engine's missing sums)
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1, 32));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2, 32));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1, 32));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2, 32));
        const int act_a = m_tile * 16 + lane / 4;
        const int act_b = act_a + 8;
        if (lane % 4 == 0) {
            scr_max[act_a * N_SUBS + sub] = mx_a;
            scr_max[act_b * N_SUBS + sub] = mx_b;
        }
        __syncthreads();

        float sn[16], ef[16];
        for (int r = 0; r < 16; ++r) {
            const int act = m_tile * 16 + r;
            float comb = scr_max[act * N_SUBS + 0];
            for (int s2 = 1; s2 < N_SUBS; ++s2) comb = fmaxf(comb, scr_max[act * N_SUBS + s2]);
            const float old_max = row_max[act];
            const float new_max = fmaxf(old_max, comb);
            sn[r] = new_max > -1e30f ? new_max : 0.0f;
            ef[r] = expf(fmaxf(old_max - sn[r], -80.0f));
        }
        // (O rescale omitted: register-only, trivially linear)
        for (int l = 0; l < 8; ++l) {
            S[l] = expf(fmaxf(S[l] - sn[fi(l)], -80.0f));
        }

        float sum_a = 0.f, sum_b = 0.f;
        sum_a += S[0] + S[1] + S[4] + S[5];
        sum_b += S[2] + S[3] + S[6] + S[7];
        sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 1, 32);
        sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 2, 32);
        sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 1, 32);
        sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 2, 32);
        if (lane % 4 == 0) {
            scr_sum[act_a * N_SUBS + sub] = sum_a;
            scr_sum[act_b * N_SUBS + sub] = sum_b;
        }
        __syncthreads();
        if (w_id % WARP_M == 0 && lane % 4 == 0) {
            for (int r = 0; r < 16; ++r) {
                const int act = m_tile * 16 + r;
                float tot = 0.f;
                for (int s2 = 0; s2 < N_SUBS; ++s2) tot += scr_sum[act * N_SUBS + s2];
                row_sum[act] = ef[r] * row_sum[act] + tot;
                row_max[act] = sn[r];
                out[64 + t * 64 + act] = row_sum[act];          // per-tile published sum
                out[64 + t * 64 + 32 + act] = tot;              // per-tile tot
                if (r == 0) {
                    for (int s2 = 0; s2 < N_SUBS; ++s2) {
                        out[256 + m_tile * 64 + t * 16 + s2 * 4] = scr_sum[act * 4 + s2];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 32; ++i) {
            out[i] = row_max[i];
            out[32 + i] = row_sum[i];
        }
    }
}

int main() {
    float * d = nullptr;
    cudaMalloc(&d, sizeof(float) * 400);
    probe6<<<dim3(1), dim3(32, 8)>>>(d);
    float h[400] = {0};
    cudaMemcpy(h, d, sizeof(float) * 400, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // host reference: online softmax over 3 tiles x 64 cols
    int bad_max = 0, bad_sum = 0;
    for (int act = 0; act < 32; ++act) {
        float mx = -1e30f;
        for (int col = 0; col < TILES * 64; ++col) mx = fmaxf(mx, synth(act, col));
        float m = -1e30f, s = 0.f;
        for (int t = 0; t < TILES; ++t) {
            float tm = -1e30f;
            for (int c = t * 64; c < (t + 1) * 64; ++c) tm = fmaxf(tm, synth(act, c));
            const float nm = fmaxf(m, tm);
            const float ef = expf(m - nm);
            float add = 0.f;
            for (int c = t * 64; c < (t + 1) * 64; ++c) add += expf(synth(act, c) - nm);
            s = ef * s + add;
            m = nm;
        }
        const float em = fabsf(h[act] - mx) > 1e-3f;
        const float es = fabsf(h[32 + act] - s) > 1e-2f * fmaxf(1.f, fabsf(s));
        if (em) { bad_max++; printf("max act=%d dev=%.4f gpu=%.4f ref=%.4f\n", act, h[act] - mx, h[act], mx); }
        if (es) { bad_sum++; printf("sum act=%d dev=%.4f gpu=%.4f ref=%.4f\n", act, h[32 + act] - s, h[32 + act], s); }
    }
    for (int t = 0; t < TILES; ++t) {
        printf("act0 tile%d: pub_sum=%.4f tot=%.4f | act16: pub=%.4f tot=%.4f\n",
            t, h[64 + t * 64 + 0], h[64 + t * 64 + 32 + 0],
               h[64 + t * 64 + 16], h[64 + t * 64 + 32 + 16]);
    }
    for (int t = 0; t < TILES; ++t) {
        printf("row0 slots t%d: %.4f %.4f %.4f %.4f | row16: %.4f %.4f %.4f %.4f\n", t,
            h[256 + t * 16], h[256 + t * 16 + 4], h[256 + t * 16 + 8], h[256 + t * 16 + 12],
            h[320 + t * 16], h[320 + t * 16 + 4], h[320 + t * 16 + 8], h[320 + t * 16 + 12]);
    }
    printf("%s max %d bad, sum %d bad (32 rows)\n", (bad_max + bad_sum) ? "FAIL" : "PASS", bad_max, bad_sum);
    return 0;
}
