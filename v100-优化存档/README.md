# V100 (SM70) llama.cpp 性能存档

| 项 | 值 |
|---|---|
| 机器 | 192.168.7.3, `Tesla V100-PCIE-32GB` (72 SM, PG503-216), 32 GB |
| 模型 | `~/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` (14.62 GiB, 27.32 B) |
| 构建 | `~/llama-cpp-sm70` @ `793dc79a2` |
| 对照 | 全部走上游 FA 内核（fork 内核已于 2026-09-26 删除）|

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
llama-server -m <model> -ngl 99 -c <ctx> -b 2048 -ub 2048 --spec-type draft-mtp
```

`--spec-type draft-mtp` 见第 2d 节（+44% decode）。输出**逐字节相同**
（`llama-server`，temp 0，seed 1234，ub=512 与 ub=2048 的生成文本完全一致），
所以这个改动是数值安全的。

### ⚠️ 2026-09-26 起：Volta-only 构建的默认值已是 2048

如果构建时 `-DCMAKE_CUDA_ARCHITECTURES=70`（**只含 70**），根 `CMakeLists.txt` 会定义
`LLAMA_VOLTA_ONLY_BUILD`，于是：

- `common/common.h` 的 `n_ubatch` 默认值变成 **2048**（不再是 512）
- `llama-bench` 的默认值同步跟进（它有自己的默认值，不读 `common_params`）
- **不需要传任何参数**

实测（llama-server，15001 token）：

| 配置 | pp tok/s |
|---|---|
| 不传任何 ub 参数（新默认生效）| **919.5** |
| 显式 `-ub 512`（仍可覆盖回旧行为）| 711.4 |
| 显式 `-ub 2048` | 910.8 |

构建时门控（两机验证）：

| 构建 | `-ub` 默认 | Volta-only 消息 |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES=70`（V100）| **2048** | 有 |
| `CMAKE_CUDA_ARCHITECTURES=75`（2070）| 512 | 无 |

**2070 的生产构建一个字节都没变。** 为什么用构建时门控而不是运行期查架构：ggml 不暴露 cc
（`ggml_backend_dev_*` 只给 name/description/memory，`ggml-cuda.h` 也没有），运行期门控要新增
公开 API 或按设备名匹配，构建时门控零成本。

> 下表中「默认」那一行的含义自此改变：在 sm_70 构建上它现在等于 2048。表里的数字是改动前测的。

### 最少只需要一个开关：`-ub 2048`

`n_ubatch = min(n_batch, n_ubatch)`（`src/llama-context.cpp:281`），而 `-b` 默认就是 2048，
所以 `-ub 2048` 单独就够（实测 1049.3，与 `-b 2048 -ub 2048` 的 1049.6 等效）：

| 配置 | pp2048 |
|---|---|
| 默认 | 795.4 |
| **`-ub 2048`** | **1049.3** |
| `-b 2048 -ub 2048` | 1049.6 |
| `-ub 4096` | 1048.8（被截到 2048）|
| `-b 4096` | 795.6（**没用**，ub 还是 512）|

⚠️ 部署脚本里若把 `-b` 调小于 2048（例如 `-b 1024`），`-ub 2048` 会被截到 1024，收益打折。
**保持 `-b >= 2048`。**

最小改动：一个 flag，或一个环境变量 `LLAMA_ARG_UBATCH=2048`。

### 三个开关都能传（已实测等效）

`-b` / `-ub` 是普通的运行时参数（`common/arg.cpp:1626-1639`），不是写死的。
llama-server，15001 token prompt：

| 途径 | 形式 | pp tok/s |
|---|---|---|
| 默认（代码里的值）| — | 712.1 |
| 命令行 | `-b 2048 -ub 2048` | **909.2** |
| **环境变量** | `LLAMA_ARG_BATCH=2048 LLAMA_ARG_UBATCH=2048` | **910.7** |

**部署时用环境变量就不必改命令行**（服务单元里加两行即可）。

⚠️ **但环境变量在 `llama-bench` 上不生效**（实测 791.78，等于默认值）：llama-bench 会用自己的
默认值覆盖 `params.n_batch/n_ubatch`。所以**测的时候用命令行，部署的时候两种都行** ——
否则拿 llama-bench 去验环境变量会误以为没效果。

### 真正的缺口：默认值写死且不看架构

```cpp
// common/common.h:468-469
int32_t n_batch  = 2048;
int32_t n_ubatch =  512;   // <- 就是它
```

全代码里没有任何按架构分支（没有 `if (cc == VOLTA) n_ubatch = 2048;`）。所以收益**存在但隐形** ——
只有知道这件事的人才会去传，换个人或换套部署脚本就退回 712 tok/s。要根治得动代码：

| 方向 | 做法 | 评价 |
|---|---|---|
| A. 按架构给默认值 | context 创建时已知设备 cc，Volta 就把 `n_ubatch` 提上去 | 见效快，但是打补丁 |
| B. 根治 | 别重复解压权重（在 ubatch 之间复用解压结果），或把 `MMQ_DP4A_MAX_BATCH_SIZE` 阈值改成按架构判断 | 更对，可上游 |

### 适用范围：这条只对 sm70 生效

`should_use_mmq`（`mmq.cu:330-356`）对 NVIDIA 的判断链：

```cpp
if (turing_mma_available(cc)) return true;                    // sm_75+ 一律 mmq（融合）
...
if (GGML_CUDA_CC_IS_NVIDIA(cc))
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;  // 64
```

| 架构 | fp16 张量核 | turing mma | prefill 路径 |
|---|---|---|---|
| Pascal (sm_60/61) | 无 | 无 | `!fp16_mma` = true -> **mmq（融合）** |
| **Volta (sm_70)** | 有 | 无 | **`ne11 < 64`** -> cuBLAS + **每块重新解压** <- 唯一中招 |
| Turing 起 (sm_75+) | 有 | 有 | **mmq（融合）** |

**Volta 是唯一落进那个阈值分支的 NVIDIA 架构**，所以不要把这个调参照搬到 2070。
（另：Pascal 也走 mmq，因为 `!fp16_mma_hardware_available` 为真。）

decode 的 +8% 机制**未查清**（decode 只有 1 个 token，ub 理论上无关）。
两次独立测量都是 +7~8%，可复现，但不要当成已解释的现象。

---

## 3. TurboQuant KV 在 V100 上是容量功能，不是速度功能

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


## 4. 投机解码：接受率决定一切，而真正的路是 MTP

**先注意一个坑**：`-md <draft>` **不会**启用投机，它只设模型路径。
`common_params_speculative::types` 默认是 `{NONE}`，只能由 `--spec-type <name>`
或 draft 仓库里的 sidecar 填充。类型名是 `draft-simple`。
不加 `--spec-type` 时服务器会打印 `[spec] loading draft model` 并占掉 3 GB 显存，
但响应里没有 `draft_n` 字段、速度也完全不变 —— 看上去像「投机无效」，
实际是「投机根本没跑」。

| 场景 | 接受率 | tg tok/s | vs 基线 |
|---|---|---|---|
| 基线（散文） | — | 37.05 | — |
| 投机 n_max=8（散文） | **24.3%** | 16.51 | **-55.4%** |
| 投机 n_max=16（散文） | — | 13.59 | -63.3% |
| 投机 n_max=8（数字列表） | **100%** | 54.06 | +45.9% |

n_max=8 时每步成本是 8 次 draft 前向 + 1 次批量验证。2B Q8 每 token 读 ~2.7 GB，
27B 读 ~14.6 GB，即 draft 约便宜 5 倍，所以盈亏平衡点大约在 **60~70% 接受率**。
实测散文只有 24.3% -> 大亏。

列表那个 100% **不是真实工作负载信号**（draft 能精确预测确定性数列）。
不要用列表/计数类 prompt 去评估投机。

### 真正的线索：这个模型自带 MTP 头

GGUF 元数据：`qwen35.nextn_predict_layers = 1`。

MTP 头是针对目标模型自己的分布训的，真实文本上的接受率应该远高于外挂 2B draft 的 24.3%。
llama.cpp 有对应的投机类型（`COMMON_SPECULATIVE_TYPE_DRAFT_MTP`，`common/common.h:174`）。
**这是尚未测过的 decode 杠杆，而且用的是模型自己的头，不是外挂模型。**
原始日志见 `speculative-2026-09-26.log`。

---

## 5. MTP 投机：+44% decode，且输出无损

模型自带 `nextn_predict_layers = 1`，llama.cpp 对应 `--spec-type draft-mtp`。
**不需要第二个模型文件**，只多占 552 MiB 显存（日志：`creating MTP draft context
against the target model`）。

| 配置 | tg tok/s | 接受率 | 输出 |
|---|---|---|---|
| 基线 | 37.03 | — | 参照 |
| **`--spec-type draft-mtp`（默认深度 3）** | **53.44** | **54.8%**（247/451）| 相同 |
| 外挂 2B draft（`draft-simple`） | 16.51 | 24.3% | — |

**+44.3%。** 对比外挂模型的 24.3% 接受率 / -55%：**MTP 头是针对目标模型自己分布训的，
这就是 +44% 与 -55% 的差别。**

### 深度不要调大

| `--spec-draft-n-max` | tg | 接受率 | 输出 vs 基线 |
|---|---|---|---|
| **默认 3** | **53.44** | 54.8% | 相同 |
| 4 | 51.31 | 44.8% | 相同 |
| 8 | 30.47 | 23.8% | **不同** ⚠️ |
| 16 | 23.15 | 12.8% | 相同 |

接受率随深度快速衰减，起草超过 ~3 个就是纯亏。

### ⚠️ 两个必须记住的坑

1. **n_max=8 时输出与基线不同。** temp 0 下投机本应无损，而 4/16/adaptive 都逐字节相同。
   最可能是批量验证（一次验 n+1 个 token）改变了 GEMM 形状 -> 累加顺序变化 ->
   近平分处 argmax 翻转。所以「投机无损」**不能假设**，每个配置都要做文本对拍。
2. **`ngram-mod` / `ngram-cache` 完全没启用**（响应里无 `draft_n`，速度等于基线）。

### 完整推荐配置（同一 job 内对比）

```
llama-server -m <model> -ngl 99 -c <ctx> -b 2048 -ub 2048 --spec-type draft-mtp
```

| | 旧默认 `-b 512 -ub 512` | 推荐配置 | 收益 |
|---|---|---|---|
| pp2048 | 755 | 1034 | +37% |
| pp16384 | 706 | 860 | +22% |
| pp65536 | 469 | 564 | +20% |
| decode（短上下文） | 37.0 | **53.4** | **+44%** |
| decode（32K 上下文） | 27.2 | 32.8 | +21% |
| 输出 | — | — | **逐字节相同** |

原始日志见 `mtp-2026-09-26.log`。

---

## 6. ⚠️ 可复现性警告（跨 job 的绝对值会飘）

同一配置（p32768 n128 r2, f16 KV, ub=2048）在不同 job 里：
pp32768 = **707.77 +- 9.94** vs **773.41 +- 2.87**；tg128 = **27.16** vs **32.75**。
差异达 ~10%。

1Cat 自己的台账记录 V100 时钟在持续负载下从 1530 MHz 掉到 1260-1275 MHz，很可能就是这个原因。

**所以：只有同一 job 内的对比可信。** 本存档所有结论性数字都取自包含对比双方的那个 job。

---

## 7. 负结果

| 开关 | 结果 | 结论 |
|---|---|---|
| `GGML_CUDA_FORCE_MMQ=1` | ub=2048 下 +0.5~1.3%；ub=512 下 **-5.2%** | 不值得全局开。mmq 的 GEMM 本身比 cuBLAS 慢，只有解压省下的比它多才划算，所以它依赖「分块很多」 |
| `GGML_CUDA_FUSE_CHAIN=0` | -2% | 默认开着是对的 |
| fork FA 内核 | 已实测后删除：V100 pp2048 -3.4% / pp16384 -20.9% / pp65536 -37.1% | 见 `fa-default-ab-2026-09-26.log`。代码已删，勿重建 |
| turbo3 KV | 32K 持平；64K prefill -5.3% / decode -6.1% | 容量功能，不是速度功能。见第 3 节 |
| 外挂 draft 投机解码 | 散文（接受率 24.3%）**-55%**；数字列表（接受率 100%）+46% | 接受率决定正负号。外挂 2B draft 与目标分布不匹配，在真实文本上是实亏。见第 4 节 |
| `ngram-mod` / `ngram-cache` | 未启用（响应无 `draft_n`，速度等于基线） | 见第 4 节 |

---
## 8. 测量方法教训

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

## 9. 复现

```bash
ssh -i ~/.ssh/id_rsa lbp@192.168.7.3
cd ~/llama-cpp-sm70
# 存档脚本见 bench-ubatch-v100.sh（在本机 Mac 侧同步过去）
bash v100-优化存档/bench-ubatch-v100.sh
```

原始输出见 `ubatch-2026-09-26.log`。
