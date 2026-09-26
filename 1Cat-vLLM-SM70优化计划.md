# 1Cat-vLLM 单卡 V100 32G 优化计划

> 定位：**针对 V100/SM70 硬件的性能工程计划**（非贡献流程）
> 硬件：单卡 Tesla V100 32GB（SXM2，HBM2 900 GB/s）
> 代码基线：1CatAI/1Cat-vLLM main (b711d5304)

> [!WARNING]
> **本文件是第一次侦察的产物，其中三个数字后来被实测推翻，保留原文仅供追溯。**
> 引用前请先读这一节：
>
> 1. **「SM 数 80」错。** 这台是 `Tesla PG503-216`，**72 SM** 的 GV100。
> 2. **「FP16 Tensor Core 125 TFLOPS」错。** 72 SM 的未降频上限是
>    `72 x 8 x 64 x 2 x 1.53 GHz = 112.8 TF/s`；持续负载下降频到 1260-1275 MHz 后
>    约 **93.4 TF/s**，cuBLAS 大矩阵实测最好 **83.45 TF/s**。这台卡跑不到 125。
> 3. **「Flash-Attention-V100：17.9 -> 60.8 TFLOP/s」口径不对。** 同项目的台账
>    `1Cat-vLLM/docs/design/sm70_fa2_d256_prefill_pipeline.md` 把两个口径分列：
>    8K 处是 **38.0 Causal TFLOP/s**（硬件真跑）与 **76.0 Full-square logical**
>    （被 mask 掉的上三角，原文：`it is not hardware-executed work`）。60.8/76.0
>    是 logical 口径，不是真实算力。
>
> 另外该台账自己的 nsys 分类显示，64K TP4 prefill 里**注意力只占 4.19%**，
> TurboMind AWQ GEMM 占 **68.17%**，并且它的 Closed Paths 第一条就写着
> 「Further attention-only work has a small current-model ceiling. Future prefill
> work should first target AWQ GEMM and elementwise overhead.」
>
> 结论：**不要把 60.8 当成可争取的提速空间，也不要按 125 TFLOPS 估收益。**
> 相关实测与更正见 `Volta-Turing-FA进度.md` 的「V100 证据复核」节。

---

## 1. V100(SM70) 硬件特征与两道性能墙

| 参数 | 数值 | 对优化的含义 |
|------|------|--------------|
| SM 数 | 80 | 与后世代相比短波前、少 L2 |
| FP16 Tensor Core | 125 TFLOPS (HMMA.884) | prefill 计算墙 |
| FP32 | 15.7 TFLOPS | 任何 FP32 路径都是灾难 |
| INT8 tensor core | 无 | FP8/Int8 矩阵只能靠拆包到 FP16 TC 或 MMQ |
| BF16 | 无 | 模型一律转 fp16 计算 |
| HBM2 | 900 GB/s | decode 带宽墙 |
| L2 | 6 MB | 远小于 Ampere 之后的 40-50MB，权重循环复用空间极小 |
| 单卡显存 | 32 GB | 27B NVFP4(~14GB) + 16GB KV 预算 |

**两道墙就是优化地图：**

```
decode   = 带宽受限 : tok/s <= 900GB/s / (每token读的字节数)
prefill  = 计算受限 : 有效TFLOP/s <= 125 (HMMA.884峰值)
```

以 Qwen3.8-27B-NVFP4 为例（64 层 = 16 full-attn + 48 GDN；KV 4 heads x 256；~14GB 权重）：

- **短上下文 decode**：每步读 ≈ 权重 14GB + KV 小头 → 天花板 ≈ 900/14.3 ≈ **62 tok/s**
- **长上下文 decode**：128K ctx 时 KV 读取并不可忽略（见第 3 节 L1 的托底算账），
  KV 字节/token 直接决定长上下文到顶速度

项目参考机 4x16G 在 128K/256K 的实测（61.8/50.4 tok/s）已经接近各自带宽墙，
说明 decode 通道本身已压满——**再想快只能减少每 token 读的字节数，或投机解码提高有效 tok/s**。

---

## 2. 项目现状盘点：已优化 vs 开放缺口

### 已落地（不用你再做）
- Flash-Attention-V100：17.9 -> 60.8 TFLOP/s（生产），79 TFLOP/s（实验，PR #315，未转默认）
- E4M3 KV + FP32 attention 数学契约（DFlash2 验证路径）
- QSA/GDN 稀疏结构集成（Qwen3.8 的 48 层线性注意力）
- DFlash2 / MTP4 / adaptive lookup q16 投机（316 tok/s 是 opt-in 记录）
- DeepSeek-V4：MXFP4 experts、MLA sparse、TP8 分层 allreduce
- 多卡通信栈：push/hierarchical allreduce（TP2/4/8 专用）
- TurboQuant **backend 已注册**（`turboquant_k8v4 / 4bit_nc / k3v4_nc / 3bit_nc`，
  Triton store+decode 内核）+ 质量 harness `benchmark_sm70_turboquant_quality.py`

### 开放缺口（= 你的机会，按价值排序）
1. **TurboQuant KV 在 SM70 的速度从未被测量/调优**：只有质量 harness，
   没有 decode 速度 benchmark；backend 优先级表里 Volta 没有 TURBOQUANT 位置
2. **79 TFLOP/s 注意力实验未转生产默认**
3. **单卡（TP1）路径无人调**：全公司成果都挂在 TP2/4/8 上，单卡无通信开销但
   也没有配套优化
4. **DV4 风格路线在单卡 32G 未验证**（8x16G 才跑的动 DeepSeek-V4）

---

## 3. 优化杠杆清单（按 ROI 排序）

### L1. TurboQuant 2-4bit KV 打到 SM70 decode 路径（头号项目）

**背景**：V100 无 BF16/int8 TC，KV 用 fp16 每 token 每层写 4KB（16 层 full-attn =
64KB/token，GDN 层另有压缩态）。E4M3 只能减一半。TurboQuant（WHT 旋转 + 3.25bit
~码）理论上是 E4M3 的 2.5x 压缩。

**现状差距**：backend 有 Triton 实现（`triton_turboquant_store/decode`），质量
harness 在，但：
- 无 SM70 速度基线（没人跑过 decode tok/s）
- Triton decode 内核大概率不接 WHT 反旋转进 FLASH_ATTN_V100 的 native 路径，
  与 60.8 TFLOP/s 的组排内核无关
- 存储路径（WHT 旋转 + 打包）是否走 SM70 友好的 HMMA.884 未知

**分三步走（每步独立可交付）**：

| 步骤 | 内容 | 交付物 | 验收 |
|------|------|--------|------|
| L1a | 在 V100 上跑通 `--kv-cache-dtype` 的 turbo 类型 + 质量 harness，测 decode tok/s 基线 | 一份"TurboQuant on V100"速度/质量对照表（vs E4M3） | 数字上墙，若速度劣于 E4M3 也是有效结论 |
| L1b | 用 ncu 拆 Triton decode 与 store 的 HBM 流量/指令瓶颈，决定"调 Triton"还是"并进 FLASH_ATTN_V100 native" | 瓶颈分析报告 + 路线决策 | 有数据支撑的选型 |
| L1c | 实现（二选一）：Triton 内核优化 或 FLASH_ATTN_V100 读 turbo-KV + 反 WHT | 合入的 decode 路径 | 长上下文 decode tok/s 超过 E4M3 同配置 ≥15%，质量 harness 过 |

**预期收益（托底算账，单卡 128K ctx）**：KV 读取从 E4M3 的
~128KB/token 级降到 ~40KB/token 级，长上下文 decode 有 +30-80% 的头部空间。
**风险**：Triton 在 SM70 上性能差、WHT 反旋转多 2 次矩阵乘、质量损失需 harness 把关。
**联动**：codec 本体（`ggml/src/ggml-turbo-quant.c`）就在你手边的
`~/Projects/llama-cpp-sm70`，两边质量门禁（turbo3 MSE=0/Cosine=1.0）可直接借用。

### L2. 79 TFLOP/s 注意力 -> 生产默认（prefill 提速）

**现状**：60.8 已生产；PR #315 做到 ~79 TFLOP/s（实验）。79/125 ≈ 63% 峰值，
仍有 37% 理论空间在 split-k 调度、寄存器驻留概率、QK/PV 管线里。

**工作**：阅读 #315 与 `bm64/native_bm32` 系列 benchmark 的差异，验证 79T 在
单卡 32G 长 prefill（Q8192+）下数值契约（FP32 accumulation）不受损，然后转默认
路线。
**交付**：prefill tok/s 对照表 + 质量门（`benchmark_sm70_attention_exactness.py`）。
**收益**：长 prefill（首 token 延迟）可再提 ~25%。
**风险**：79T 是窄配置（特定 head 数/长度），转默认需要泛化验证。

### L3. 单卡 TP1 decode 路径审计

**背景**：fork 里大量优化带着 TP 假设（push allreduce、hierarchical allreduce、
TP2 专用 GDN 投影拼接、TP4 MTP5）。单卡时这些路径要么空转要么没被压过。

**工作**：
- 用 `benchmark_sm70_decode.py` + `benchmark_sm70_lm_head_top1/20.py` 建立单卡矩阵
- ncu 审计单卡 decode 的 HBM 流量构成（权重 / KV / 激活 / 中间）
- 找出"TP 时代妥协"：比如带 allreduce 依赖的共享权重缓存、图上多余的集合通信
**交付**：单卡 decode 流量分解表 + 1-2 个消除浪费的 PR。
**收益**：短上下文 decode 从"接近墙"再挤 5-15%；更重要的是让单卡成为可信测量平台。

### L4. spark 注意力与投机在单卡的组合

- DFlash2（单卡可用，MTP4 需要 TP 不可用）已接通：优化重点是 drafter 权重驻留
  与 rejection 路径（已有 `benchmark_sm70_dflash2_*` 群可复测）
- adaptive lookup q16（316 tok/s 的 opt-in 组件）评估在单卡长上下文是否成立
- QSA/GDN 的稀疏收益在单卡 KV 预算下最明显：KV 预算 16GB 时稀疏结构 = 更长的有效上下文

### L5. 权重侧确认（预期低 ROI，做一遍验证就行）

NVFP4 shared-weights 缓存已做（`[Kernel] Cache small shared NVFP4 output weights`）。
更低 bit（MXFP3）质量风险高；权重侧的带宽墙（14GB/token）靠投机绕，不靠压缩绕。
结论预计：权重侧保持 NVFP4，不投入。

---

## 4. 执行路线图

```
P0  (现在,无GPU)  建立单卡 roofline 模型与测量脚本设计
P1  (V100就绪)    单卡基线矩阵：E4M3 vs turboquant 4bit/3bit vs fp16,
                  prefill/decode/long-context 三轴 + 质量 harness
P2  (L1a 完成)    TurboQuant on V100 速度/质量报告 -> 决策 L1b/L1c
P3  (L1c)         TurboQuant decode 落地 SM70 native 路径 (头号交付)
P4  (L2)          79T 注意力实验 -> 生产默认
P5  (L3+L4)       单卡 decode 审计优化 + DFlash2 单卡调优
P6  (可选)        vLLM 侧质量门与 llama-cpp-sm70 TurboQuant 码本对齐
```

每个 P 的验收 = 实测数字 + 质量门，缺一不可（项目惯例）。

---

## 5. 测量方法学（先在 P0 定死，避免后面各说各话）

1. **带宽墙公式**：`tok/s_ceiling = 900 / bytes_per_token`。每步先在
   ncu 里把 `dram__bytes_read.sum` / `dram__bytes_write.sum` 列出来对账。
2. **三轴报告**：配置矩阵固定为（模型, 长度, batch, KV dtype, attention route）。
   单卡数字永远注明 "1x V100-32G"，不许和 4x16G 表混比。
3. **质量门**：沿用 `benchmark_sm70_turboquant_quality.py`（top-logprob 分布 +
   长输出 canary），再叠加 DFlash2 的 FP32-state 契约。
4. **对照顺序**：先 E4M3（现有生产），后 turbo3/turbo4，最后才谈发散调参。

---

## 6. 与 llama-cpp-sm70（你的另一个 repo）的协作方式

| 协作点 | llama-cpp-sm70 (TurboQuant codec) | 1Cat-vLLM |
|--------|-----------------------------------|-----------|
| 码本/WHT 常量 | `ggml-turbo-quant.c` 是唯一真理, 保持 byte-identical | 只消费, 不重造 |
| 质量门禁 | turbo3 MSE=0/Cosine=1.0, turbo4 Cosine=0.9956 | 对齐到 vLLM 侧的 top-logprob 分布 |
| CPU 参考反 WHT | 现成可当数值金标准 | Triton/native 内核用它对拍 |
| 复用策略 | WHT 旋转本身就是 128x128 正交矩阵乘, 两端共享 | SM70 native 反旋转直接照搬 |

方向：codec 保持上游一致，vLLM 侧只做"读 turbo layout 的 SM70 内核 + 反旋转 +
质量验证"。这样两个仓库的 KV 格式永远互通。

---

## 7. 一句话总结

你的优化主线应该是：**L1 把 TurboQuant 2-4bit KV 变成 SM70 decode 的生产路径**
——它同时是你两个仓库的交叉点、单卡长上下文的头号带宽杠杆、而且现状是"有后端、
有质量门、无人测速度"的明确空白。先从 P0 的 roofline 与测量脚本开始，V100 恢复后
第一个上午就能产出 L1a 的速度/质量对照表。