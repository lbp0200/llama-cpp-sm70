# 1Cat Flash-V100 移植到 llama.cpp —— 进度与 TODO

> 更新：2026-09-24（2070 正确性修复完成: Turing frag 映射 + v2 截断）
> 分支：`feature/v100-fa-port`（5 commits，本地已提交）
>   491ecdd04 graph null-deref 修复
>   57f551ae7 v2 截断（旧实现，已被 02b7ffa8d 取代）
>   06837a189 Turing fragment 映射修复 + cfg 拆分 + gate 750
>   02b7ffa8d v2 截断（修复后重实现）
> 机器：V100 = 192.168.7.3（sm_70, 32G）；2.16 = 10.1.2.16 RTX 2070（sm_75）

---

## 当前代码状态（3 commits，feature/v100-fa-port）

| commit | 内容 | 状态 |
|--------|------|------|
| `803ad9904` | **v1 Volta FA 内核**（split-D、fp32 smem、子块 softmax、sinks、D512 stub、门控 GGML_V100_FA） | ✅ 验证过（正确性 + 性能基准） |
| `491ecdd04` | **CUDA graph null-deref 修复**（`node->src[0]` 空检查，ARANGE/FILL/DIAG 等无输入 op） | ✅ 全量验证 |
| `57f551ae7` | **v2 kv_max 截断**（side kernel 预计算可见上界 + 主内核查表） | ⚠️ 已提交但实测异常（见下），运行态已回滚到 803ad9904 |

## 已完成的验证（可靠数据）

### 2.16 (2070/sm_75)
- 全量 `test-backend-ops` **19643/19643 passed, 0 segfault**（默认 CUDA graph 开启）
- 证书存档：Mac `sm75-验证存档/`（4 份日志）
- 修复的两个 fork 既有缺陷：
  1. **D512 VEC 编译缺陷**（sm_70/75 ptxas smem 超限）——没有修复连全量编译都不行
  2. **CUDA graph null-deref**（无输入 op 一 graph capture 就崩，gdb 定位 ggml-cuda.cu:2928）

### V100 (sm_70)
- 构建/同步：**OK**（CUDA 12.9）
- ARANGE + graph 修复在 sm_70：**通过**
- FA 全集（除 hsk=320 已知缺陷）：**7747/7747**（hsk=320 permute launch invalid-argument 是 fork 第 3 个 sm_70 既有缺陷，未修）
- **A/B**（Qwen3.8-27B-IQ4_XS, f16 KV）：
  - 新内核 v1：pp2048 **768.9** / pp65536 **299.5** / decode 36.0
  - 旧路径：pp2048 **799.3** / pp65536 **465.5** / decode 37.4
  - v1 短上下文 -3.8%、**长上下文 -36%（主因：无因果截断 + KV 重复读）**

## v2 截断实验（回滚）

- 方案：host 预生成 per-row 可见上界（side kernel `fattn_v100_mask_kvmax`，每 query 行一个 block 扫 mask）→ 主内核查表 clamp num_n_tiles
- 快速验证：105/105（kv=1024 子集）+ pp2048 实跑 778.8 tok/s 正常
- **异常**：pp65536 bench 19 分钟未完成；单 op FLASH_ATTN_EXT kv=16384 也 240s 卡（非干扰，180W 在算即真算但极慢）
- **v2 待调试的两个实验**（下一次直接做）：
  1. `GGML_CUDA_DISABLE_GRAPHS=1` + v2 → 跑 kv=16384 单 op：**快 = graph 交互问题；慢 = 内核问题**
  2. 若 graph 相关：怀疑每 chunk side kernel + pool 分配在 capture/重新捕获流程中退化

## 教训（8 小时踩坑记录，下次必查）

1. **V100 生产 llama-server 曾开机自启**（占 25G + 100% GPU）——**现已 `systemctl disable`**，重启不会回来
2. **ssh 超时断连 → 远程进程残留** → 显存被占 → 后续全部"failed to load / 卡死"假象。**跑长任务先清残留**：
   ```bash
   for p in $(pgrep -f llama-bench); do kill -9 $p; done   # 勿用 pkill -f（会杀 ssh 自己）
   for p in $(pgrep -f test-backend-ops); do kill -9 $p; done
   ```
3. 多次 kill/graph 异常后 GPU 可能被污染 → reboot 也未必能靠 nvidia-smi 恢复（本次重启后正常）
4. **不要多轮小编辑拼 kernel 结构**（本次 v2 第一次实现 3-4 轮编辑出死锁；第二次一次性写了才干净）

## 2070 (SM75) 适配进度（已完成，2026-09-24）

### 状态：正确性完全修复，已提交
- 提交 `06837a189`：Turing fragment 映射修复 + cfg 拆分 + gate 750 + 防御
- 提交 `02b7ffa8d`：v2 截断（mask-derived KV scan bound）
- 验证：`FLASH_ATTN_EXT` hsk=256 **470/470**；hsk 64..640 **7764/7764**；
  真模型 translategemma-4b 输出 `Hello, world.`（与旧路径一致）

### 根因（重要，通用教训）
**1Cat 的 WMMA fragment -> (row, col) 展开公式只适用于 Volta（HMMA.884）。**
Turing 用 HMMA.16816，fragment 元素顺序不同，于是 mask/scale 被加到错误的
(row, col) 上：首行 query 的输出被破坏，自回归下整句生成全错。

定位路径（供以后参考）：
1. test-backend-ops 的**随机 mask** case 只暴露「M=0 行 1.8% 元素 0.05-0.10 偏差」
2. **真模型对拍**（因果 mask）暴露灾难性错误（输出空/其它语言短语）
3. gmem dump（kernel printf 在 graph capture 下不可靠，必须用 device buffer + host 读回）
   - QK/softmax/PV/写址/写落地 全部逐段验证正确
   - mask 数据正确、mask 读法正确、valid_q_rows 正确
4. `probe4.cu`（2070 单 warp 最小复现）**直接证明**：按 1Cat 公式给 (row,col)=(0,0)
   加 marker，store 后出现错位（sS[0][2] 也有值、行 2/3 整行错）

### 修复方式（架构无关）
QK 阶段**只 store 原始分数**；`scale + mask` 在 `store_matrix_sync` 之后
**于 smem 上按显式 (row, col) 应用**（全线程遍历 valid_q_rows x BLOCK_N）。
代价是每 tile 一次 smem 读写（16x32 fp32 = 2KB），可忽略。

### 性能（2070, translategemma-4b, f16 KV）
| | v1 内核 | v2 截断 | 旧路径 |
|---|---|---|---|
| pp512 | 2707 | 2916 | 3113 |
| pp2048 | 2262 | 2403 | 3086 |
| tg32 | 96.2 | 106.1 | 107.9 |

- v2 在 v1 基础上 +6~10%（截断生效），但仍略落后旧路径
- **原因**：gemma3-4B 的 pp 瓶颈是权重读（FA 占比小），4B 级模型不足以
  体现 FA 优化；**FA 优化的价值在大模型 + 长上下文（V100 27B 场景）**
- 2070 的成果是**正确的 SM75 内核 + 通用 frag 映射修复**，性能验证应回 V100

### 调试设施（保留，env `GGML_V100_DEBUG=1` 启用，生产零开销）
- launcher: pool 分配 128 float debug buffer，host 端 `fprintf` 打印
- kernel: QK 后 sS 行/列统计、mask 直读对照、PV/softmax 值
- `/tmp/probe4.cu`（2070）：fragment 映射最小复现，跨 arch 验证用

## V100 (sm_70) 回归验证结果（2026-09-24 已执行，机器上线当日）

### 结论：修复后全绿，无 sm_70 回归
| 项 | 结果 |
|---|---|
| hsk=256 连跑 4 轮 | **4x 470/470**（修复 6716ea4a9 后；修复前 3 轮中 2 轮 469） |
| 全量 sweep hsk=(64\|128\|192\|256\|512\|576\|640) | **7747/7747**（排除已知缺陷 hsk=320） |
| 旧路径对照 GGML_V100_FA=0 连跑 3 轮 | 3x 470/470（证明抖动是新路径回归，非环境） |
| 真模型文本冒烟（Qwen3.8-27B） | 输出连贯正确（推理正确给出 Paris） |

### 真模型 A/B（Qwen3.8-27B IQ4_XS，f16 KV，llama-bench）
| | 新内核 v2 | 旧路径 | 差距 | 历史 v1 |
|---|---|---|---|---|
| pp2048 | **774.6** | 757.5 | **+2.3%** | 768.9（-4%） |
| pp65536 | 286.6 | **461.8** | **-38%** | 299.5（-36%） |
| tg32 | 36.0 | 25.9* | +39%* | 36.0 |

\* 旧路径 tg 本轮方差异常大（±3.48，历史值 37.4），需复测；新内核 36.0 稳定。

### tg32 基线复测（2026-09-24，异常值已替换）
| | 3 次实测 | 均值 | 运行内方差 |
|---|---|---|---|
| 旧路径 GGML_V100_FA=0 | 37.41 / 37.43 / 37.43 | **37.42** | ±0.26~0.28 |
| 新内核 | 36.96 / 36.95 / 36.98 | **36.96** | ±0.27~0.28 |

- 旧路径 ±3.48/25.87 判定为**单次异常**（疑当时 GPU 状态污染），作废
- 真实 tg 差距：**-1.2%（36.96 vs 37.42），基本持平**（上表 "+39%" 一列作废）
- 结论：decode 无优化空间争议，攻坚目标只有 pp65536

### 关键分析：v2 截断对 causal prefill 长上下文无效（重要认知修正）
- prefill 的 query 都是**最新的 token**（第 c 个 chunk 的行可见 0..N_past+行号），
  每行的可见上限都 ≈ N（整个 KV），只有 chunk 内部的"未来列"可跳：
  M/N = 512/65536 ≈ **0.8%** —— 所以 pp65536 没有任何改善（286 vs v1 299）
- **-38% 的真因不是缺截断**。初步假设（待验证）：BLOCK_M=32 vs 旧路径 64，
  K/V tile 复用差一倍 -> KV 读流量 x2，N 越大越吃亏（pp2048 反而 +2% 与此吻合：
  短 N 时计算主导、长 N 时带宽主导）
- 下一步方向（按预期收益排序）：
  1. 增大 BLOCK_M 或跨 q-tile 共享 K/V 加载（治 KV 流量）
  2. GQA 组内共享 KV 读（qwen 27B GQA ratio 8 -> KV 读 /8）
  3. 复测旧路径 tg（方差 ±3.48 可疑）

### 仍保留的原验证清单（供下次回归参考）

改动对 cfg_v100 的影响已逐项审计（cfg<32,64,512> 参数一字未改；frag 修复数学等价；
safe_max 只在全屏蔽行不同；gate 新增检查在 V100 上冗余；v2 截断的 mask 语义审计一致）。
剩余风险只在「精确参数组合」，必须用 V100 实测确认：

```bash
# 0. 构建
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build -j24

# 1. FA 正确性：期望 470/470（含 hsk=320 的已知 fork 缺陷 case 会 abort，见下）
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256"        # 470/470
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=(64|128|192|256|512|576|640)"  # 7277/7277

# 2. 真模型 A/B（Qwen3.8-27B IQ4_XS, f16 KV）
./build/bin/llama-bench -m <27B> -p 2048,65536 -n 32 -ngl 99 -ctk f16 -ctv f16
GGML_V100_FA=0 ./build/bin/llama-bench -m <27B> -p 2048,65536 -n 32 -ngl 99 -ctk f16 -ctv f16
# 期望：frag 修复 + v2 截断后，长上下文差距从 -36% 显著缩小（甚至反超）

# 3. 回退开关（若任何异常）
GGML_V100_FA=0    # 立即回到旧路径
```

## 已知缺陷（非本次引入，勿误判为回归）

### 1. sm_70 上 hsk=320 的 permute FA case abort
- 现象：`FLASH_ATTN_EXT(hsk=320,hsv=256,...,permute=[0,2,1,3],kv_view=1,...)`
  → `CUDA error: invalid argument`，abort 在 `ggml-cuda.cu:113`
- 复现（V100）：
  ```bash
  ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=320"
  GGML_V100_FA=0 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=320"   # 同样复现
  ```
- 性质：**fork 既有缺陷**（与本次 Volta FA 内核无关，关掉新内核仍复现）
- 影响：FA 全量 sweep 在 hsk=320 处中断；hsk=320 之外全部正常
  （实测 V100：hsk 64..640 排除 320 后 7747/7747 全过；2070 无此缺陷，含 320 全量 7764/7764）
- 相关：上游 issue ggml-org/llama.cpp#28037（Volta FA 路径问题）上下文
- TODO：定位 launch 参数（320 头维 + 非连续 view 的组合），确认是否值得单独修/上报

### 2. ~~FATTN 随机精度抖动~~ —— 已撤销：真身是 v2 截断聚合 bug（已修 6716ea4a9）
- 误判经过：初判为「random_device 播种导致的阈值边缘抖动、非回归」，
  **错了** —— 旧路径 `GGML_V100_FA=0` 连跑 3 轮全部 470/470 稳定，
  抖动只出现在新路径（3 轮中 2 轮 469），是本 fork 的回归
- 真根因：v2 截断的 `kv_ceiling` 只取块内**最后一行**的可见上限。
  causal/SWA mask 可见性随行递增，取最后一行 == 取 max（真实负载恰好正确）；
  但 test 的 `init_tensor_kq_mask` 是**任意随机可见性**（均匀随机值 + 随机
  128x64 块打 -inf，约 10% 像素），前一行可能看到最后一行看不到的列，
  这些列被错误跳过 -> ERR≈9.4e-4（略超 5e-4 阈值）
- 为何时好时坏：随机 mask 决定「坏切口」是否出现，单轮 470 case 里
  P(至少一个坏块) 随机 -> 交替 470/469
- 修复（`6716ea4a9`）：ceiling 改为**块内所有行取 max** —— 只跳过所有行都
  看不到的列，对任意 mask 形状都精确
- 验证：V100 修复后 **4 轮 x 470/470** + 全量 **7747/7747**；
  旧路径 3 轮 470/470（排除环境因素）
- 教训：「边缘 ERR + 随机数据」不能直接归因为抖动，必须跑旧路径 A/B 才能定性

## pp65536 -38% 根因诊断（2026-09-24 task-2，受控实验完成）

### N 扫描曲线（llama-bench -p 2048,8192,16384,32768,65536 -r2，f16 KV）
| N | OLD | NEW | NEW/OLD 速度比 | 隐含注意力时间占比 s |
|---|---|---|---|---|
| 2048 | 796.0 | 733.5 | 0.92 | 8.5% |
| 8192 | 760.4 | 609.1 | 0.80 | 24.8% |
| 16384 | 696.0 | 520.9 | 0.75 | 33.6% |
| 32768 | 579.3 | 411.7 | 0.71 | 40.7% |
| 65536 | 470.7 | 287.5 | 0.61 | 63.7% |

拟合：`t_new/t_old = 1 + s(N) x (R-1)`，**R=2.0 一个值打满全部 5 点**；
64K 处 s=64% 与 FLOP 估算（attn core 4ND vs linear ~813MFLOP/tok -> ~65%）吻合。

### 确认的主机制：旧路径 GQA x2 KV 打包 -> 我们 KV 流量恰为 2 倍
- 本模型 24 Q 头 / 4 KV 头 -> **gqa=6**，D=256，64 层
- 旧 mma 内核 Volta 分支（fattn.cu switch_ncols2）：gqa%2==0 -> **ncols2=2**：
  一个块算 32 token x 2 个 Q 头（ncols1=32, ncols=64 列），K/V 读一遍共享
  -> 每 KV 头 KV 流 = (M/32) x (24/2) = **48**
- 我们 grid=(M/32, 1, H)，每块 1 头扫全 KV -> 每 KV 头 = **96 流 = 2.0x**
- 代码证据：`ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`（Volta:
  8/4/2 档）+ `iter_z_gqa = ceil(gqa/ncols2)` + `zt_Q = z_KV*gqa + zt_gqa*ncols2`

### 排除清单（全部有据）
1. **稀疏路径**：`may_use_sparse` 仅配 D=512/576（fattn-mma-f16.cuh:2048），
   D=256 永不走 -> 排除
2. **KV_max 截断/稀疏触发**：llama-graph.cpp 所有 build_attn_mha 调用点
   n_kv_max 实参恒为 **0**（除 MoE top_k）-> 旧路径无截断、shall_use_sparse=false
3. **per-token causal 截断机制本身**：prefill 的 query 全是最新的 token，
   每行可见上限 ~= N，可跳的只有 chunk 内未来列 M/N ~= 0.8% -> 两条路径都
   无从截断（这同时解释 v2 截断为何对 pp65536 无效）-> 排除
4. **side-kernel 开销**：全 mask 扫描 M x N x 2B vs 主流量 (M/32)xH x N x 1024B
   = 0.26%，launch 61 层 x 128 chunk x ~3us ~ 23ms / 229s = 0.01% -> 解析排除
5. **mask-on-smem 代价**：每 tile 16KB smem RMW ~2-3ns vs ~87ns/tile ~3% -> 排除
6. **grid 调度/L2 复用假说**：若 L2 能吸收我们多出的流，2048 点速度比应 ->1，
   实测 0.92 仍符合 R=2 模型 -> L2 未抹平差距，重排 grid 预测收益 ~0 -> 不选

### task-3 选型（按证据）
**把每 KV 头的 96 流砍到 48（匹配旧路径）**，两条等价路线取简：
- 选：**BLOCK_M 32 -> 64（单头）**：块数 (M/64)xH = 96/2 = 每 KV 头 48 流，
  mask/dst/行映射零改动，纯 BM 变化
- 配套：**O 搬回寄存器**（WMMA C fragment 常驻跨 n-tile，去 per-tile smem 累加；
  末尾 store_fragment -> 复用已死的 S/P smem -> dst），否则 BM=64 的 O=64KB
  爆 smem（96KB 上限）。新 smem: Q32+S16+P8+KV16 = 72KB <= 96KB
- 预期：注意力 89s -> ~45s，pp65536 ~400-450（合同线 330 之上有余量）
- 后续增强（不在本任务）：GQA x3 打包（rows=96, Q 也进寄存器）-> 64 流，再 -33%

## task-3 实施记录：KV 流量削减 + 同步瘦身（2026-09-24 凌晨）

### 落地改动（feature/v100-fa-port 工作区）
1. **BLOCK_M 32 -> 64**（每 KV 头 96 -> 48 流，匹配旧 mma GQA x2 打包的流量）
2. **O 常驻寄存器（REGS_O）**：WMMA C fragment 跨 n-tile 常驻（BM64 的 fp32 O 需 64KB smem，放不下）
   - 重缩放走 **warp 私有 16x16 scratch**：store -> scale -> load 全程 warp 局部，**零 barrier**
   - final 归一化直接经私有 scratch 写 dst，**零 barrier**
3. **mask 内联进 QK**：每 warp 对自己的 16x16 tile 原位 scale+mask（省 1 sync/tile），尾列直写 -inf
4. K/V split（去 union）+ 曾试软件流水预取（实测中性，最终保留内联加载）
5. BN 64->32（split 后 smem 预算所迫：85KB<96KB）

### 调试史诗（一次真实根修）
- 433/470 ERR~1.2 起点 -> 逐段取证：dst=O/sum 自洽、sS=prem*scale+mask 自洽、shfl 每行独立、
  P=exp(sS-max) 末 tile 精确、轨迹 16 tile sum 链手算全对、单 tile 模式 16/16 列 == 独立重算
- **根因：`store_matrix_sync(sOS + tm * 64 ...)` 把 tile 索引(0..3)当行偏移**，应为 `tm * WMMA_M * 64`：
  band0(tm=0) 恰好自洽 -> 行0 幸存、其余 band 错位互踩 -> "只有 row0 正确 + 散列损坏"全部解释
- 修复后 470/470、全量 7747/7747、真模型正确

### 被实验证伪的优化假说（重要 negative results）
| 假说 | 实验 | 结果 |
|---|---|---|
| KV 流量减半 -> 长上下文大幅收益 | BM64 落地 | pp65536 无增益（L2 已把 distinct bytes 吸收，流数不是瓶颈） |
| 载入-计算串行是旧路径优势 | split-KV + K/V 软件流水预取 | 中性（259->262 反向波动内） |
| 重缩放轮 barrier 是主开销 | warp 私有 scratch 零同步 | +10（252->261）小幅 |
| mask-close sync 可省 | mask 内联 QK | 平坦（261->259/262） |
| BN=64 减半 tile 固定开销 | 实测 | 更慢（当时含分轮 bug），后因 smem 放弃 |

**当前定论**：BM64/BN32/split/regs 版与 HEAD(v2 smem-O) 同场 A/B 见下表；旧路径的
1.6x 优势未被上述任何单一机制解释，遗留假说 = 旧 mma 的 nbatch_fa 分段细粒度 CTA
（grid 远大于输出 tile 数、seam fixup）+ cp.async 多级流水，列为后续课题。

### 同场三方 A/B（Qwen3.8-27B IQ4_XS f16 KV, r2，06:00-06:25 同会话）
| | pp512 | pp2048 | pp65536 | tg32 (r3) |
|---|---|---|---|---|
| HEAD（v2 mask-trunc, BM32/BN64 smem-O） | 784.3 | 774.7 | **288.9** | 37.0* |
| NEW 实验（BM64/BN32 REGS_O inline-mask） | 777.9 | 761.9 | **261.3 (-9.5%)** | 29.5 |
| 旧路径 GGML_V100_FA=0 | 781.3 | 783.4 | **468.4** | 34.2 |

\* tg HEAD 值取自昨晨同配置复测 36.96/37.0；本会话 tg 整体方差大（OLD 也从37.4漂到34.2）。

### 结论（决定不合入）
1. **BM64/REGS_O 路线对 pp65536 是 -9.5% 实回归、tg 也回退 -> 实验以 `781861f2e` 留档，
   `fb7ee08b3` 回滚到已验证内核**。合入态 = 门禁全绿的 HEAD 内核（4x470 + 7747 + 冒烟）。
2. 任务合同的 pp65536>=330 未达成（HEAD 自身 288.9，需要在 HEAD 基础上 +14% 才达标）；
   本夜证伪了所有预设杠杆（见上表），**旧路径 1.6x 的真因仍未定位**。
3. 遗留假说修正（已读 `fattn-mma-f16.cuh` 配置表，一实一伪）：
   - **Volta 对 (256,256,64) 无专属配置**（`get_config_volta` 只特判 D=512/576/640，
     其余 `// TODO tune specifically for Volta` 落回 ampere 表）：
     `nthreads=512, occupancy=1, nbatch_fa=64, nbatch_K2=128, nbatch_V2=128,
     nbatch_combine=64, nstages_target=1, Q_in_reg=true`
   - **伪**：cp.async 多级流水 —— `nstages_target=1`（定义注释：1 == 永远同步加载）
     ⇒ 旧路径 D=256 并不流水化，我们的串行加载不是差距来源。
   - **实（精化）**：`nbatch_fa=64` ⇒ `iter_k = ceil(N/64)`，旧路径 grid 是
     **(N/64) x (M/32) x (gqa/2) x H_KV ≈ 20 万个细粒度 CTA**（seam fixup 组合部分和），
     每 CTA 只碰 64 个 KV 位置（工作集极小、延迟靠 CTA 级并行隐藏），而我们是
     192 个 CTA 各自扫全 N。外加 `Q_in_reg=true`（Q 常驻寄存器，零 smem Q 流量，
     我们每 k-tile 都从 smem 载入 Q fragment）。
   - **下一步方向**：把 v1 内核改为 nbatch_fa 分段 CTA（块内只扫64列 + 部分和 fixup）
     或等价的 KV 分段 grid 重构；Q_in_reg 次之。或在上游 issue #28037 语境下
     与维护者对齐优化路线。

## 2070 回线验证（task-6，条件跳过，2026-09-24 07:24 记录）

- 07:17:17 窗口过后共 4 次 SSH 探测（07:18:30 一次 + 07:19~07:24 三次，
  ConnectTimeout 12-15s）全部 `Connection timed out` —— 2070 未回线。
- 按 task-6 合同条款跳过，不阻塞目标。
- **回线后手动补做清单**：
  ```bash
  cd ~/llama-cpp-sm70 && git pull    # 至 9a2a876d3+
  ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256"   # 期望 470/470
  ./build/bin/llama-cli -m ~/models/translategemma-4b-it.i1-Q4_K_M.gguf \
      -p "Translate to English: Bonjour le monde." -n 50 --no-jinja  # -> Hello world.
  ```

## 下一步 TODO（按优先级）

### T-A. v2 调试（根因定位）
```
① 上机：v2 版本（57f551ae7 的 fattn-v100.cuh）
② GGML_CUDA_DISABLE_GRAPHS=1 跑 FLASH_ATTN_EXT kv=16384 单 op 计时
   - 快 → graph 交互（side kernel/alloc 在 capture）→ 改分配方式/合并扫描进主 kernel
   - 慢 → 内核 → 从 kv_max clamp 逻辑二分
③ 修好后重跑 pp65536 A/B（目标：299 → 反超 465）
```

### T-B. v3 设计（GQA 头合并打包）
- 背景：新内核每 block 一个 Q head → **KV 读重复 gqa_ratio=6 次**（24Q/4KV）；旧路径 mma 有 ncols 打包
- 估计：65536 时 KV 读 8GB×6=50GB vs 最小 8GB，是长上下文另一个大头的差距
- 结构：grid.z 改按 **KV head**，块内循环 gqa_ratio 个 Q head 共享同一批 KV smem tile（QK/PV 扩展 ncols）
- 风险：需在 test FATTN 的 GQA case（nh=8/nr23=[4,1] 等）重新验证

### T-C. 后续计划（原有）
- 128K 长上下文数值对拍（T3）
- FULL/PIECEWISE graph 思路（T4，vLLM-2080Ti 参考）
- hsk=320 sm_70 缺陷记录上报 fork

## 关键实现备忘

- **门控**：`ggml_cuda_flash_attn_ext_v100_available()`——cc==700 + D=256 + f16 K/V + f32 Q/out + mask 存在 + ne[3]==1 + max_bias/softcap==0；`GGML_V100_FA=0` 回退
- **遮盖**：FA 输出 dst = permute(0,2,1,3) `[D][H][M][B]`：token 用 nb[2]、head 用 nb[1]（大坑，勿再写反）
- **smem 92KB**：cudaFuncSetAttribute opt-in
- sm_70/75 D512 VEC stub：CUDA 上从不 dispatch，零语义影响
- 调试工具（2.16 上 /tmp）：probe1/2/3（QK/softmax/PV 微测）、fa_ab.cpp（CPU/CUDA 后端 A/B 微测）