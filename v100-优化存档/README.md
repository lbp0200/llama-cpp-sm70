# V100 (SM70) llama.cpp 性能存档

| 项 | 值 |
|---|---|
| 机器 | 192.168.7.3, `Tesla V100-PCIE-32GB` (72 SM, PG503-216), 32 GB |
| 模型 | `~/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` (14.62 GiB, 27.32 B) |
| 构建 | `~/llama-cpp-sm70` @ `793dc79a2` |
| 对照 | 全部 `GGML_V100_FA=0` (上游 FA 内核 = V100 上的部署选择) |

---

## 1. prefill 有 20-32% 花在「重复解压权重」

`nsys profile --stats=true` 的内核时间分布。pp65536 的两次 pass 共 263.7 s 内核时间，
对 2 x 135.7 s 的 wall，即内核占 wall 的 97%；pp2048 内核占 wall 的 85%。

| 阶段 | pp2048 | pp65536 |
|---|---|---|
| `flash_attn_ext_f16` | 2.5% | **39.0%** |
| `cutlass ..._s884gemm_f16_128x128` | **47.9%** | 29.9% |
| `dequantize_block_iq4_xs` | 27.5% | 17.2% |
| `dequantize_block_q5_K` | 4.2% | 2.6% |
| `gated_delta_net_cuda` | 7.5% | 4.7% |
| `convert_unary<float,__half>` | 2.4% | 1.5% |

**解压权重合计 31.7% (pp2048) / 19.8% (pp65536)。**

### 根因

```cpp
// ggml/src/ggml-cuda/mmq.cuh:8
#define MMQ_DP4A_MAX_BATCH_SIZE 64

// ggml/src/ggml-cuda/mmq.cu:356
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
```

V100 有 FP16 张量核，所以 prefill (`ne11 >= 64`) 一律走
「先把权重解压成 f16，再 cuBLAS」。解压**每个 ubatch 重来一遍**，
开销 = `ceil(prompt / ub) x 权重字节数`，因此与上下文长度成正比。

`n_ubatch` 默认 512 (`common/common.h:469`)，`n_ubatch = min(n_batch, n_ubatch)`
(`src/llama-context.cpp:281`)。

---

## 2. `-ub 2048` 是免费收益

`llama-bench -ngl 99`，r=3（pp65536 为 r=2）：

| 场景 | `-ub 512` (默认) | `-ub 2048` | 收益 |
|---|---|---|---|
| pp2048 | 755.5 +- 3.3 | **1008.2 +- 20.4** | **+33.4%** |
| pp16384 | 706.2 +- 3.8 | **867.7 +- 35.8** | **+22.9%** |
| pp65536 | 469.2 +- 0.8 | **563.6 +- 1.9** | **+20.1%** |
| tg128 | 33.71 +- 0.75 | **36.42 +- 0.91** | **+8.0%** |

ub 的最优值是 **2048**，再大更慢（`-b` 与 `-ub` 一起调）：

| `-b` / `-ub` | pp16384 |
|---|---|
| 2048 / 2048 | **867.7** |
| 4096 / 4096 | 841.7 |
| 8192 / 8192 | 817.5 |

### `llama-server` 复核（部署形态）

15001 token prompt，同一进程内 3 次：

| 配置 | 3 次实测 | 均值 |
|---|---|---|
| `-b 512 -ub 512` | 704.2 / 699.8 / 680.0 | 694.7 |
| `-b 2048 -ub 2048` | 884.9 / 829.6 / 807.5 | **840.7** |

**+21.0%**，三对全部同向。

### 推荐配置

```
llama-server -m <model> -ngl 99 -c <ctx> -b 2048 -ub 2048
```

输出**逐字节相同**（`llama-server`，temp 0，seed 1234，ub=512 与 ub=2048 的生成文本完全一致），
所以这个改动是数值安全的。

decode 的 +8% 机制**未查清**（decode 只有 1 个 token，ub 理论上无关）。
两次独立测量都是 +7~8%，可复现，但不要当成已解释的现象。

---

## 2b. TurboQuant KV 在 V100 上是容量功能，不是速度功能

`llama-bench -p <ctx> -n 128`，ub=2048。注意 `-ctk turbo3 -ctv turbo3` 会被 fork 的
`TURBO_AUTO_ASYMMETRIC=1` 自动改成 `q8_0` K + `turbo3` V（该模型 GQA=6）。

| KV | pp32768 (r=2) | tg128 (r=2) |
|---|---|---|
| f16 / f16 | 707.8 +- 9.9 | 27.16 +- 2.24 |
| q8_0 / q8_0 | 699.9 +- 3.0 | 27.02 +- 2.12 |
| q8_0 / turbo3 | 707.1 +- 7.5 | 26.31 +- 1.93 |
| q8_0 / turbo4 | 701.8 +- 6.3 | 26.79 +- 2.07 |

| KV | pp65536 (r=1) | tg128 (r=1) |
|---|---|---|
| f16 / f16 | **596.1** | **28.16** |
| q8_0 / turbo3 | 564.5 | 26.45 |

**32K 全在 1 sigma 内（无收益），64K 是 prefill -5.3% / decode -6.1% 的实回归。**

### 为什么

decode 在 32K 是 27 tok/s = 37 ms/token，而工作集是 14.6 GB 权重 + 2 GB KV = 16.6 GB
-> 有效带宽只有 **451 GB/s**（峰值约 900）。**decode 是延迟/占用率受限，不是字节受限**，
所以省 KV 字节买不到任何东西，而 turbo 路径反而在注意力内核里加了反量化计算。

这个模型的 KV 每 token（64 层中 16 层全注意力，4 个 KV 头，D=256）：
`4 x 256 x 2 x 2 B x 16 = 64 KiB`（f16），turbo3 约 13 KiB。
即 KV 占 decode 工作集的 12%（32K）/ 23%（64K）。

**这是本 fork 的招牌功能，在 V100 上不给速度。** 原始日志见 `kv-2026-09-26.log`。

---

## 3. 负结果

| 开关 | 结果 | 结论 |
|---|---|---|
| `GGML_CUDA_FORCE_MMQ=1` | ub=2048 下 +0.5~1.3%；ub=512 下 **-5.2%** | 不值得全局开。mmq 的 GEMM 本身比 cuBLAS 慢，只有解压省下的比它多才划算，所以它依赖「分块很多」 |
| `GGML_CUDA_FUSE_CHAIN=0` | -2% | 默认开着是对的 |
| `GGML_V100_FA=1` (fork FA) | pp65536 -17~-18% | 见 `Volta-Turing-FA进度.md`，V100 默认应为 OFF |
| turbo3 KV | 32K 持平；64K prefill -5.3% / decode -6.1% | 容量功能，不是速度功能。见 2b |

---

## 4. 测量方法教训

1. **r=1 不可信，而且系统性偏低。** 同一配置 (`pp16384 / ub=2048`) 三次测量：
   r=1 得 850.93 / 810.47；r=3 得 861.12 +- 8.11 / 867.68 +- 35.80。
   所有 r=1 的数字都要打折，不能用来下结论。
2. **不要在 GPU 上并行跑两个 bench。** 一次 job 因争用直接失败；
   另一次在被污染的卡上给出 850.93 这种偏高值。
3. **`pgrep -f llama-bench` 会杀掉 ssh 自己**（远程命令行里含 "llama-bench"，
   被 `-f` 全命令行匹配命中）。用 `pkill -x llama-bench`。
4. **nsys 默认 trace buffer 对大量小内核不够。** decode 那次只抓到 ~47 ms 内核
   （实际 3.8 s），prefill 那次是完整的。decode 的内核分解需要调大 buffer 重做。

---

## 5. 复现

```bash
ssh -i ~/.ssh/id_rsa lbp@192.168.7.3
cd ~/llama-cpp-sm70
# 存档脚本见 bench-ubatch-v100.sh（在本机 Mac 侧同步过去）
bash v100-优化存档/bench-ubatch-v100.sh
```

原始输出见 `ubatch-2026-09-26.log`。
