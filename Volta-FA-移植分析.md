# 移植 1Cat Flash-V100 Attention 到 llama.cpp —— 源码级分析与方案

> 结论先行：**比"直接搬文件"复杂，但比"换整个 vLLM runtime"简单得多。**
> 1Cat 没有任何一个 .cu 能直接 include 进 llama.cpp（torch ABI + paged KV + 裸
> `__half` 指针），但它的**数据流设计**全部可移植，而且 llama.cpp 自己的 KV 布局
> 是连续（非 paged）的，适配层反而比 vLLM 侧简单。
> 另：llama.cpp 在 sm_70 上**已经有** WMMA 的 mma-f16 路径（含 Volta GQA 打包），
> 所以这不是"从零移植 FA"，而是"用 1Cat 的结构升级现有路径"。

---

## 1. 双方现状核实（源码级）

### 1.1 llama.cpp 侧（本项目）

| 文件 | 角色 |
|------|------|
| `fattn.cu` | 宿主 dispatch；**sm_70 有专门分支**（L204）：GQA 打包按 %8/%4/%2 切 `mma_f16_switch_ncols1`，volta 用取模（ncols=8/4/2/1），非 volta 用阈值比较 |
| `fattn-mma-f16.cuh` | WMMA m16n16k16、f32 累加、QK/PV 融合流水；D=256 配置 nthreads=128-256，nbatch_fa=8..64 Q行，nbatch_K2=64..128 token |
| `fattn-mma-turbo.cuh` | 本项目 fork 新增：turbo2/3/4 KV 的 MMA 路径（D 128/256，turing MMA 门控） |
| `fattn-vec.cuh` | 非张量核回退（量化 KV 等） |
| `fattn-common.cuh / fattn-swizzle.cuh / fattn-tile.cuh` | smem tile 加载器、swizzle、量化反量化进 smem |
| `fattn.cuh` | 公共接口 |

sm_70 现状：**有 WMMA 路径 + Volta GQA dispatch + turbo KV 支持**。这说明
你的 650~700 tok/s 不是"没有 FA"，而是"现有 FA 的数据流在 V100 上不够好"。

### 1.2 1Cat 侧（flash-attention-v100/）

| 文件 | 行数 | 角色 | 对移植的意义 |
|------|------|------|--------------|
| `kernel/fused_mha_forward.cu` | 1124 | **非 paged prefill（最接近 llama.cpp 布局）** | 主移植源 |
| `kernel/fused_mha_forward_paged.cu` | 4094 | paged prefill（vLLM 专用） | 只提取思想 |
| `kernel/flash_decode_paged.cu` | 6276 | paged decode/XQA | 第二期 |
| `kernel/flash_decode_turboquant.cu` | 574 | turbo KV decode | 与本 fork turbo 类型对拍 |
| `kernel/fused_mha_backward.cu` | 1353 | 训练反向 | **不需要（推理）** |
| `kernel/fp8_kv_bridge.cu` | 249 | fp8 KV 桥 | 可后置 |
| `kernel/h3/` | - | H3/state kernels | 不需要 |
| `kernel/flash_v100_traits.cuh` | 118 | 块尺寸/布局常量 | 移植核心参数 |

**1Cat prefill 的关键结构（已核实）**：
- D=256：`BLOCK_M=32`（Q 行）x `BLOCK_N=64`（token/块），**16 warps = 512 线程/块**
- 全程 fp32 累加（QK、softmax、PV），softmax 用 `RCP_LN2` log2 域技巧（D<256），
  D=256 走显式 fp32 scale + `P_SUB_TILE` 分段 softmax；LSE 显式写 fp32
- `__ldg` 读 only 缓存 + `uint4` 向量化加载（Volta 无 cp.async，用 L1 只读路径）
- 同一个块内 QK 与 PV 的 tile 级并行（16 warps 分摊 QK/PV tile）
- 与项目 docs 一致："FP32 accumulation for SM70 long prefill"（128K 上下文数值正确性）

### 1.3 结构差异总结（1Cat 相对 llama.cpp sm70 路径的德尔塔）

| 维度 | llama.cpp sm70 现状 | 1Cat | 影响 |
|------|---------------------|------|------|
| 每块线程 | 128-256（4-8 warps） | 512（16 warps） | 更长上下文下 L2/带宽利用率 |
| Q行/块 | nbatch_fa=8..64 | 32 | 平衡点不同 |
| token/块 | nbatch_K2=64..128 | 64 | 1Cat 用小 N 块换更多并行块 |
| KV tile 加载 | 自有 swizzle loader | uint4 `__ldg` + 相位交错 K layout | 可在 llama.cpp loader 上复刻 |
| softmax | 在线（fragment 内） | 显式 P_SUB_TILE + fp32 LSE | 128K 长上下文的数值差异 |
| GQA 打包 | ncols=8 上限（Volta 取模） | README 声称 6-head 打包宽 QK/PV | 你的模型 GQA=6（24Q/4KV），打包策略直接相关 |
| 双缓冲/预取 | 有（nstages 配置） | K tile prefetch + PV double buffer | 1Cat 方案更明确 |

---

## 1.4 上游 llama.cpp#28037 交叉验证（2026-09-21 复查）

上游 issue（`ggml-org/llama.cpp` 28037，作者 hernandez42）与本文结论高度一致，且多给出一个实测数据点。本地代码逐条核实：

| #28037 的关键论断 | 本地代码核实 | 一致性 |
|---|---|---|
| `ggml_cuda_fattn_mma_get_config_volta()` 回退到 Ampere 配置 | `fattn-mma-f16.cuh:125` 只有 D=512/576/640 的特例，D=256 走 `return ggml_cuda_fattn_mma_get_config_ampere(...)`，注释 `// TODO tune specifically for Volta` | 完全一致 |
| 调参无效（当前设计已近最优） | 其 SM72 实测：自定义 Volta 配置 decode 20.85->19.68 t/s（归还 smem 压力换 occupancy） | 接受该结论：**决定性红利只能靠结构重写** |
| Volta 无 cp.async（无流水重叠） | `fattn-mma-f16.cuh:381` `cp_async_available(cc) && ... ? nstages : 0`，Volta 恒为 0；设备侧#ifdef 同样闭合 | 完全一致，这是 1Cat K-prefetch/PV 双缓冲价值的具体切入点 |
| 建议路径 1：Volta split-D WMMA（仿 1Cat）| 与本文 L1 方案同一设计；其估测 SM72 上约 +35% | 思路一致，数值不可直接迁移（见下） |
| 建议路径 2/3：MTP heads、persistent blocks | 与本 fork turbo MMA 正交；对用户是 M4 之后的二期项 | MTP 与你的"MTP decode ~20 tok/s"场景相关，但不是 attention 主路线 |

**对计划的三个新增结论：**

1. **该 issue 无主（OPEN，0 comments，作者等维护者兴趣）**。在你的 fork 里做 A/B，
   成功后合入上游 PR 关闭 #28037，是一条有据可查的贡献路径（AGENTS.md 的
   "明确的需求缺口"由此满足）。
2. **其估测 +35% 不能直接搬到 V100 27B 128K prefill**：他们在 Jetson AGX Xavier
   （SM72，8 SM 级别、CUDA 11.4）上用 Qwen3.6-35B-A3B MoE，而你是 V100
   （80 SM、125 TFLOPS TC）+ 27B + 长 prefill，负载构成完全不同。+35% 是
   kernel 级粗估，你的真实收益仍然以第 4 节的 attention 占比表为准。
3. split-D 的理由现在可以写死：D=256 拆 4x64 时每 warp 的 smem 从 ~32KB 降到
   ~8KB（块内 16 warp 才能装得下），同时把 D 方向的序列化 WMMA 链拆成 4 路并行。

---

## 1.5 SM75（RTX 2070）扩展边界

用户另有 RTX 2070（sm_75）。能不能一起优化：**能，同一个内核模板通吃
SM70+SM72+SM75，靠 cc 分两套 config 表；但收益和角色必须分清。**

| 架构分界 | 代码证据 | 影响 |
|---|---|---|
| SM70/SM75 都无 cp.async | `common.cuh:356` `cp_async_available` = highest_compiled_arch >= AMPERE；`fattn-mma-f16.cuh:381` nstages 由此恒 0 | 1Cat 手动双缓冲对两代卡都有效 |
| SM75 有 ldmatrix | `fattn-mma-f16.cuh:932/958/1309` `load_ldmatrix`；sm_75+ | Turing 同代码自动受益 |
| smem/块上限 64KB（Turing）vs 96KB（Volta） | 架构手册 | 16-warp/BLOCK32x64 装不进 Turing，必须分 config |
| Turing 已有专门调度 | `fattn.cu:439` `cc >= GGML_CUDA_CC_TURING` 阈值式 GQA；`get_config_turing` 独立表 | 移植是升级不是新开 |

**设计：一个模板，两套 config。**

```
fattn-v100.cuh（新）
  cc == VOLTA   : 16 warp / BLOCK 32x64 / split-D 4x64 / fp32 全程
  cc == TURING  : 同算法，smem 64KB 预算下 8-12 warp / BLOCK 16x32，ldmatrix 加载
```

默认只在 SM70 启用；SM75 先 A/B 再决定是否默认。手动双缓冲 + fp32 累加 +
GQA 宽打包三点两代通用。

**2070 的定位（重要）**：
- 不是优化目标同等级：无 HMMA.884 问题（split-D 收益减半）、带宽墙 448GB/s
  （27B IQ4 decode 上限约 31 tok/s）、8GB 装不下 27B 长上下文。
- **是现成的开发/测试机**：等 V100 期间就能编译本 fork、跑 test-backend-ops
  （CUDA0 门禁）、用 7-8B 模型跑 llama-bench A/B；模型小所以改内核当天可验正确性。
- 测试数字注明硬件，不许和 V100 混比；最终 128K 数字仍在 V100 上出。

---

## 2. 哪些能直接移 / 哪些必须改

| 内容 | 判定 | 说明 |
|------|------|------|
| `fused_mha_forward.cu` 内核体 | **改造移植**（不可 include） | 依赖 torch/ATen + `__half` + vLLM ndim 语义；需换成 ggml 的 `half *`、ggml 张量步长、llama.cpp 的 `mmqa.cuh`/wmma 封装 |
| WMMA/HMMA.884 调用 | **可直接复用思路** | 两边都是 `nvcuda::wmma` m16n16k16；llama.cpp 已有 `mma.cuh` 封装，直接用 |
| 16-warp / BLOCK_MxN 布局 | **直接搬常量**（flash_v100_traits.cuh L118） | 纯数值，无依赖 |
| fp32 累加 / log2-softmax / P_SUB_TILE | **直接搬算法** | 无依赖，纯 CUDA 数学 |
| `__ldg`+uint4 加载、K 相位交错 | **搬进 llama.cpp 的 loader** | llama.cpp 加载器在 `fattn-common.cuh`/`fattn-tile.cuh`，把加载模式换掉即可 |
| paged KV / page table | **不需要** | llama.cpp KV 连续；跳过 paged 变体全部代码 |
| torch extension / setup.py / ABI | **不需要** | 只搬内核，不搬构建系统 |
| GVK/QSA/GDN/MTP/DFlash | **不需要** | 与注意力移植正交，第二期再说 |

**结论：可移植的是"内核结构 + 布局常量 + 算法"，必须适配的是"tensor 语义 + 加载器"。**
没有任何一行能原样 copy，但一半以上的"代码量"是纯 CUDA 数学，机械搬移即可。

---

## 3. 工程量估算

| 文件 | 动作 | 预估量 |
|------|------|--------|
| 新 `ggml/src/ggml-cuda/fattn-v100.cuh` | 新建：1Cat 风格 fp16 K/V、D=128/256、GQA、fp32 全程、16-warp 内核，复用 llama.cpp 的 `mma.cuh` 与 smem loader | 900-1500 行 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | 小改：给 loader 增加 uint4/`__ldg` + 相位交错变体（新增模板参数） | +200-300 行 |
| `ggml/src/ggml-cuda/fattn.cu` | 小改：sm_70 分支新增 dispatch 进入 v100 路径（现有 mma 保留作 A/B 对照） | +60-100 行 |
| `ggml/src/ggml-cuda/CMakeLists.txt` | 编译新文件 | 1 行 |
| 测试 | `test-backend-ops -o MUL_MAT` 全绿 + `test-backend-ops -o FLASH_ATTN_EXT`（如存在）+ llama-bench A/B | 无新文件 |

**第一版最小可行 patch ≈ 2 个新文件 + 2 个修改文件，~1200-1900 行。**
前提：fp16 K/V、D=256、GQA 6:1（用户模型的真实形状）、长上下文（预fill）。

---

## 4. 收益预期（诚实版）

1Cat 的 17.92 -> 60.8 TFLOP/s（3.4x）是 **attention 算子** 的提升，不是整模型。
你的 650~700 tok/s 是完整 128K prefill（projection + attention + MoE + ...）。

**收益上限 = attention 占 prefill 时间比例 x attention 提速 x 能转移到你模型的比例**

| 场景（待测量） | attention 占比 | attention 提速(假设) | 整体 prefill 提升 | 估测 tok/s |
|----------------|----------------|----------------------|-------------------|------------|
| 乐观 | 60% | 2.0x | +33% | 870-930 |
| 中性 | 40% | 1.5x | +17% | 760-820 |
| 悲观（投影/MoE 吃满） | 25% | 1.5x | +9% | 710-760 |

**在动任何内核之前，第一件事是测出 attention 的占比**（见第 6 节）。如果占比 < 25%，
移植收益撑不起 ~1900 行的工作量——那应该转去优化投影 GEMM 或 MoE，而不是 attention。

注意：你的模型 KV 是 f16 还是 q8_0/turbo 会改变走哪条路径（mma-f16 vs turbo MMA）。
第一版只做 f16 K/V，把变量锁死。

---

## 5. 关键风险

1. **数值契约**：128K 上下文下 llama.cpp 现路径的累加精度未知，需要先建立
   `llama-perplexity`/长上下文一致性基线（1Cat 为这个专门做了 FP32 契约）。
   移植后必须对拍 token 级输出。
2. **cpp 侧不做 paged**：1Cat 的 60.8 TFLOP/s 有 paged 布局的贡献，非 paged 的
   `fused_mha_forward.cu`（17.92 那条路径的升级版）未必有同等提升——所以预期按
   表格保守值走。
3. **swizzle 与 GQA 打包组合**：llama.cpp 的 Volta 走取模 GQA 打包，1Cat 是 6-head
   打包；你的 24Q/4KV 恰好 GQA=6，是 1Cat 打法的理想形状，也可能是 llama.cpp 打法的
   差形状——`%4` 在 GQA=6 时不整除，会落到 ncols=2 或 1 的次优路径。
4. **占坑冲突**：本 fork 的 turbo MMA 路径和上游 23k 条 test-backend-ops 门禁
   必须全绿（参见 AGENTS.md 测试门）。

---

## 6. 执行路线（测量先行）

```
M0（V100 恢复后, 半天） 确定现状：
  ncu 跑 128K prefill，细分 attention/投影/MoE 耗时占比
  确认当前走 mma-f16 还是 turbo MMA 路径（KV dtype 决定）
  决定：值不值得移植（占比 ≥ 25% 才继续）

M1（2-3 天）          建精确基线：
  llama-bench 128K/-ctk f16/-ctv f16 vs 现用配置
  长上下文一致性样例（perplexity + 一段 8K 生成对拍）

M2（第 1 版 patch）    fattn-v100.cuh + loader 变体 + dispatch：
  只做 fp16 K/V、D=256、GQA、长 prefill
  与 mma-f16 做 A/B（同一机器同一模型）

M3（验证）             test-backend-ops 全绿 + A/B 数字 + 数值对拍
M4（后续）             turbo KV 路径、decode 路径（flash_decode_paged 思想）、
                      fp8 KV（fp8_kv_bridge 思想）
```

---

## 7. 第一版 patch 状态（2026-09-21）

已落地的代码改动（未编译/未测试，等待有 nvcc 的机器）：

| 文件 | 改动 | 状态 |
|------|------|------|
| `ggml/src/ggml-cuda/fattn-v100.cuh` | 新建 566 行：FlashAttention-1 风格内核（16 warp，BLOCK 32x64，fp32 分数 smem，32 列子块 softmax，PV WMMA）+ 宿主 launcher + 门控 | 已写，待编译 |
| `ggml/src/ggml-cuda/fattn.cu` | include + `ggml_cuda_flash_attn_ext` 里的 Volta 门（9 行），默认开，`GGML_V100_FA=0` 回退 | 已接，待编译 |

v1 范围（有意锁死）：CUDA only、cc==700 (V100)、D=256、fp16 K/V、f32 Q/输出
（llama.cpp FA 语义）、单序列 ne[3]==1、带 f16 mask（scale 后相加，兼容 SWA）、
无 ALiBi/softcap、无 sparse、无 KV_max。不满足门控时自动回退旧路径，天然 A/B。

**待验证清单（按优先级）：**

1. **编译**：2070 (sm_75 arch370?) 或任何 CUDA 机器 `cmake -B build -DGGML_CUDA=ON`；
   关注意 smem 92416B < 96KB、launch_bounds(512,2) 无编译错误。
2. **数值正确性（最高风险点）**：
   `./build/bin/test-backend-ops -o FLASH_ATTN_EXT`（或对应 op 名）在 V100 上对拍 CPU。
   风险：WMMA fragment 的 (row,col) 映射（从 1Cat 复制的 permute）若与 nvcuda::wmma
   布局不一致，会静默面具错 token——这是必须由 test-backend-ops 抓的第一件事。
3. **A/B**：同机同模型 `llama-bench -ctk f16 -ctv f16`，`GGML_V100_FA=1` vs `=0`。
4. **长上下文一致性**：V100 上 8K 生成对拍（perplexity + token 级）。
5. **性能剖析**：ncu `dram__bytes` 对账，round-robin 拆分 attention 占 prefill 时间比例。

已知限制（后续版本再扩）：decode 不走新路径（设计如此，v1 只做 prefill）；
turbo/q8 KV 不走新路径（v2 目标）；无 stream-k（每个 block 全量游走 KV，靠
256K/64 = 2048 个 tile 与 384 block 并行补足）；mask 逐元素 gmem 读（未缓存到 smem）。

开发机建议：**2070 上先编译+跑 test-backend-ops**（数值正确性不依赖 V100），
V100 只留给最终 128K 数字。

---

## 8. 与备选路线的关系

| 路线 | 工作 | 收益 | 判定 |
|------|------|------|------|
| **移植 attention**（本文） | ~1900 行, 2 文件 | 若 attention 占比高则 +15~33% prefill | **首选**（科学干净, A/B 可做） |
| 换 1Cat-vLLM 整个 runtime | 环境迁移 + GGUF->NVFP4 重量化 + 布局重来 | 更大但依赖换模型格式 | 后手, 数据模型不同不能直接套 |
| TurboQuant KV（原计划 L1） | 另一条线 | 长上下文 KV 带宽 | 与 attention 移植正交, 可并行 |

用户在 Github 上看到的那份评估（4000 字）与本文一致的地方：移植 kernel 优于搬
runtime；不一致的地方：llama.cpp **不是**从零的 FA（已有 Volta WMMA 路径），所以
收益是"结构性升级"而非"从无到有"，预期必须按第 4 节表格保守。