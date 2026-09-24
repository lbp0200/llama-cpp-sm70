# 1Cat Flash-V100 移植到 llama.cpp —— 进度与 TODO

> 更新：2026-09-24 晚（route B mma 引擎入库 + 性能阶梯；HEAD `0dfb65ea9`，已推送）
> 分支：`feature/v100-fa-port`（origin = github.com/lbp0200/llama-cpp-sm70）
> 本文件是战役总记录：**顶部 = 现状与待办；下方按时间序保留全部历史、证据与踩坑。**
> 机器：V100 = 192.168.7.3（sm_70, 32G，**当前离线**）；2.16 = 10.1.2.16 RTX 2070（sm_75，在线，`bolt-remote`）

## 现状摘要（2026-09-24 晚）

**已交付、默认启用（主干零回归）**
- Turing pair 路径（wmma + 寄存器 O，M>=128 且 GQA 偶数）：pp1024 2978 / **pp4096 2731** / tg32 108；
  相对旧路径（3054）差距 -10.6%（历史：pad 战役 +10.9% 累计）
- V100 路径：sm_70 全绿（470/470、sweep 7747），历史 r3 契约 pp512 795 / pp2048 788 / pp65536 333 / tg32 37；
  V100 字节自 SM75 工作起未动（pad 按配置分级，硬规则）

**route B mma 引擎（正确、opt-in：`GGML_V100_FA_MMA=1`）**
- 正确性：470/470（默认路 + 显式路）、sweep 7747/7747、冒烟正常
- 性能阶梯 pp4096：1771 -> 1762（swizzle，瓶颈证伪，保留）-> 1813（删冗余 sync + ef 惰性化）
  -> 1841.5（Q_in_reg 窗口化）-> **2115.6（rescale 增长守卫，+14.9%）** -> **2171.2（comb 扫描 shfl 化，+2.6%）**；
  累计 **+22.6%**，距 wmma pair **-20.5%**

## 待办（按优先级，下个 session 直接开工）

1. ~~comb 扫描 shfl 化~~ **已完成**（本 commit；pp4096 2115.6 -> 2171.2，+2.6%；累计 +22.6%，
   差距 -20.5%；gate 470/470 ×2 + sweep 7747 + 冒烟 "Hello world." 全绿）。改法：每 lane 只算自己
   需要的 6 行（`a=lane/4`、`a+8` 已在 `mx_a/mx_b` 里且组内 4 lane 同值；O 行 `c,c+1,c+8,c+9` 用
   4 条 `__shfl_sync` 取），`grew` 只覆盖本 lane 的 O 行（正是它 rescale 的行，判据仍严格），
   publish（warp0）自行合并 16 行不再读 `sn[]`。**收益只有 +2.6% => 扫描不是大头**
2. **publish / 跨 warp softmax（下个首选）**：每 tile 有 5 次 `__syncthreads()` + scratch 两次往返，
   且 warp0 的 8 个 lane 要串行做 16 行（16 expf + 32 次 smem 读写）——这也解释了 #1 的低收益。
   旧路径 np-warp 列内 shfl、无 scratch 无 publish
   -> **设计档见文末「设计档案：publish / 跨 warp softmax 重构」**（5 个 barrier 的 hazard 表 +
   两个局部候选：publish 改 16 lane 各 1 行、V 载入提前到 publish 前；K/V union 分离已否决）
3. **Q 全 D 常驻**（16 frag = 64 regs，实测 spill）与 **launch_bounds(256,2) 对拍**
4. 目标线：先 >=2731（超 wmma pair），再 >=2900（原验收线），tg 保持 108
5. V100 回线后：**sm_70 验证 route B 引擎**（Volta C-layout 上游分支已备，但我们的 lane 版
   ldm/get_i wrapper 需在 sm70 上验证）；按用户指示恢复生产服务

## 外部参考源（2026-09-24 侦察，`~/SharedData/ninfer-v100`）

单卡 V100 的自研 C++/CUDA 推理引擎（Qwen3.8-27B：decode 峰值 219 tok/s、prefill ~1085 tok/s、
INT8 group-64 KV）。三处直接价值：

1. **`src/ops/common/volta_mma.cuh`（153 行）= V100 侧 fragment/布局参考（最有用）**：逐字转录
   llama.cpp `mma.cuh`，且**语义操作数角色按 Volta fattn-mma-f16 的调用点确定、不按 tile 形状推断**
   （正是我们 probe5 之前踩的坑）；给出 Volta 真实原语 **`mma.sync.m8n8k4`**（非 Turing 的
   m16n8k16）、K 操作数 `I_MAJOR_MIRRORED`（4 组 8 lane 复制）、Q/K/D/P/V 各角色的每线程寄存器数
   与 `get_i/get_j`，且**已对独立 host oracle 校验**。-> route B 引擎搬 sm_70（V100 回线后）时
   lane 版 wrapper 的 Volta 分支可照抄，省一轮 probe
2. **`src/ops/launcher/gqa_attention_volta_flash.cu`（611 行）= V100 长 causal prefill 的现成路由**：
   独立驱动 vendored llama.cpp `fattn-mma-f16`（pin `62bf73d25`），自带 **stream-K 分解**、按几何缓存
   config、smem 预算样板 —— 与我们实测「旧路径长 prefill 更强」（V100 pp65536 469 vs 我们 333）
   结论一致。若做 V100 混合派发（长 prefill 交给旧内核），这是现成配方；也可 diff 其 vendored 副本
3. **数值/KV 压缩对标**：INT8 group-64 KV + 对持久 K 与瞬时 Q 施加归一化 **D256 Hadamard 变换**
   （与 TurboQuant WHT 同族：他们在 FA 前旋转并压到 INT8，我们压到 turbo2/3/4 位）；
   `docs/performance/methodology.md` 的测量/发布规则可当外部标尺

边界：不含自研 softmax-FA 新内核（长 prefill 借 llama.cpp，解码侧为 m8n8k4 自研 exact Volta kernels，
与我们的 m16n8k16 不同代际）；强制 CUDA 12.8（CUDA 13 砍掉 Volta 离线编译）；V100 当前离线，(1)(2) 先读不能验。

## 铁律（每次改动必跑，本文件下半部有全部出处）

- 门禁：`test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256"` -> 470/470（默认路 + `GGML_V100_FA_MMA=1` 两路）
  ；sweep `-p "hsk=(64|128|192|256|512|576|640)"` -> 7747/7747 + 冒烟（`llama-cli`）有正常输出
- **探针前置**：引擎类改动先跑 `probe5`（原子：QK/PV 操作数方向）与 `probe6`（跨 warp 行协议），
  双 PASS 再动内核 —— 历史上这两个探针各抓到一类致命 bug（`ldmatrix`/`get_i` 的裸 `threadIdx.x`
  单 warp 陷阱；xor 归约写成赋值而非组合）
- 证据落盘：bench 全部 `tee` 进 `sm75-优化存档/` 并随 commit 入库（该目录已加 gitignore 例外）
- 远程跑脚本：**`./run-2070.sh <仓库内脚本>`**（先 rsync 同步再在机器上 `bash ... </dev/null`），
  标准门禁已脚本化：`./run-2070.sh sm75-优化存档/gate-2070.sh`。不要 `scp`（多余）也不要
  `ssh 'bash -s' < file` —— 后者会被 `llama-cli` 吃掉 stdin 里剩余的脚本行（曾整段吞掉 bench）
- 对照开关：`GGML_V100_FA_MMA=0`（pair wmma）/ `GGML_V100_FA=0`（上游 mma）随时 A/B
- ncu 不可用（`ERR_NVGPUCTRPERM`，需 root 改驱动参数并重启）-> 性能问题只能靠 A/B 阶梯二分
- **代码同步到 2070：`./sync-2070.sh`**（rsync，全树含未提交状态；排除 `build/`、`.pi/`、探针二进制；
  同步后把 2070 的 origin 固定为匿名只读 `https://github.com/...`）。**2070 不做 GitHub 认证、不放私钥**，
  推送只在 Mac 端做。历史上 gh-proxy 间歇 403 / 未跟踪文件挡 pull 的问题随 rsync 一并消失
- 2070 上 `~/.ssh/id_rsa` 是从 Mac 拷过去、且未在 GitHub 登记的旧私钥副本（2026-09-08）——已确认
  该框的 GitHub SSH 认证不可用；本工作流不需要它，建议删除（见对话记录）

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

## 旧路径内层结构解剖（08:05，fattn-mma-f16.cuh 实读，下一步选型依据）

QK 内核（本配置 Q_in_reg=true，nbatch_fa=64）：
```cpp
for (i_KQ = 0; i_KQ < nbatch_fa; i_KQ += np*T_A_KQ::I)   // KV 位置为外层
  for (k_KQ = k0; k < k0_stop; k += T_A_KQ::J) {          // D 方向内层
      load_ldmatrix(K_A, swizzled_smem);                  // K 分片经 swizzle+ldmatrix 装入
      mma(KQ_C[i/...], Q_B[...], K_A);                    // Q 分片常驻寄存器（整个循环零 Q smem 流量）
  }
```
与我们内层的结构差异（每 BN-tile 重复）：
| | 旧路径 | 我们 (WMMA) |
|---|---|---|
| 矩阵指令 | mma.sync（Volta 交换 A/B）+ ldmatrix swizzle | nvcuda::wmma load_matrix_sync |
| Q | **寄存器常驻**（Q_in_reg，每 tile 零 smem 读） | 每 (n,k) 从 sQ load_matrix |
| KQ 分数 | KQ_C 按 wide/col-major 寄存器布局（np warps 分列） | **整块物化 smem（S）**，softmax 读写 smem |
| 批处理 | KV 位置 64 一批（nbatch_fa），批间 rescale | 每 BN 整批 |
| smem swizzle | fattn-swizzle 显式 XOR（K 方向） | 无（行主序 stride 256，可能有 bank conflict） |

⟹ 两条可证伪的下一步候选（按成本排序）：
A. **给 S/K/Q 的 smem 访问加 fattn-swizzle**（复用现有基础设施，~1-2h，若 bank conflict 显著
   则立竿见影；无 ncu 无法预测，只能 A/B 实测）；
B. **Q_in_reg 移植**：Q 分片常驻寄存器（省 Q smem 读 + 释放 16KB → 或换取更大 tile），
   中等改动（QK 循环重组 + 寄存器预算）；
C. **mma.sync+ldmatrix 重写 QK/PV**（结构级，多轮次，须防 Volta mma 操作数布局陷阱——
   fragment 教训在案，动手前先单 warp probe）。

## 终局合同验证（09:0x，全绿，task-5 达成）

三连微杠杆（grid head-major + P_SUB_TILE=64 + **KV smem 行距偏斜 16 halfs**）合入后：

| llama-bench (r3) | 新内核 | 旧路径 | 差距 |
|---|---|---|---|
| pp512 | **795.4** | 784.5 | **+1.4% 反超** |
| pp2048 | **787.5** | 769.1 | **+2.4% 反超** |
| pp65536 | **333.3** | 469.5 | **-38% -> -29%** |
| tg32 | **36.9** | 30.7* | +20%（旧路径 tg 方差历来大） |

合同逐项：pp65536 333.3 ≥330 ✓ / pp2048 787.5 ≥765 ✓ / tg32 36.9 ≥35 ✓
门禁：hsk=256 x2 = 470/470 ✓，全量 7747/7747 ✓，冒烟 Paris ✓
**决定性杠杆 = KV_STRIDE 256→272 行距偏斜（298.4 -> 333.9, +11.9%）**：
WMMA 在 512B 对齐行距上的 K/V 分片加载受 bank conflict 重创——旧路径的
fattn-swizzle 一直就是在干这件事。后续可继续：Q_STRIDE/P_STRIDE 偏斜
（预算内仅容一处 +1KB，Q 优先）、S_STRIDE 偏斜（需腾 smem）、pad 尺寸调优。

## 微杠杆增量（07:30-08:00，全部落在已验证内核上）

| 杠杆 | pp65536 | 判定 |
|---|---|---|
| 基线（合入态 fb7ee08b3 = HEAD 内核） | 288.9 | — |
| + grid 改 head-major（同 kv 头发射相邻） | ~291.9 (+1%) | 保留（62b2a493d） |
| + P_SUB_TILE 32->64（单 sub，softmax barrier 减半） | **298.4 ±0.3 (+3.3%)** | 保留（62b2a493d） |
| + mask 内联进 QK（省 mask barrier） | 297.3（-0.4%） | 回退 |
| 470/470 门禁全程绿 | | |

距 task-5 合同 330 仍差 +10.6%。微杠杆时代基本结束；下一阶段按证据两条路：
1. 深读旧路径 `flash_attn_ext_f16_iter` 内层（Volta mma 形态、Q_in_reg 落法、
   nbatch_fa=64 的 rescale 批处理），量化每 tile 指令流差异后再动手；
2. 或移植 nbatch_fa 分段 CTA + dstk_fixup 部分和（结构性，多轮次）。

## SM75 pad 战役（2026-09-24 中午，用户指令：2.16 上继续 + 参考 vLLM-2080Ti-Definitive）

参考侦察结论：FlashQLA = Gated DeltaNet 线性注意力（QwenLM/weicj SM70-75 移植），
非经典 softmax-FA；其 gdn 补丁仅 shfl 技巧，对本内核数据流参考价值有限（经典
attention 走 FlashInfer，源未本地化，留待后用）。README 实战提醒：长 prefill
降频易误判为软件回退（本系列 r3 方差 ±0.4-11 正常）。

### 基线与实验矩阵（translategemma-4b f16, r3, 存档 sm75-优化存档/）
| 配置 | pp1024 | pp4096 | tg32 | 判定 |
|---|---|---|---|---|
| 基线 cf613e2bb | 2872 | 2464 | 108.0 | — |
| **+Q_PAD=8** | 2870 | **2571 (+4.4%)** | 107.8 | 采用 |
| **+P_PAD=8** | 2978 | **2697 (+4.9%)** | 108.2 | 采用 |
| +S_PAD=4 | 2990 | 2701 (中性) | 108.1 | 放弃(吃光余量) |
| cfg_sm75 BN64->32 | 2869 | **2410 (-10.6%)** | 108.8 | 否决 |
| KV_PAD 8->4 | — | — | — | **崩溃 misaligned address**（520B 行距奇 tile 未对齐；约束：pad 必须 8 的倍数保 16B 行距，已写进注释） |

### 最终配置（Q8+P8+KV8, BN64）门禁
- hsk=256 x2 = 470/470, 470/470；全量 **7747/7747**；冒烟 Hello world. ✓
- pp4096 2464 -> **~2697 (+9.4%)**，对旧路径(3054)差距 **-19.1% -> -11.7%**
- V100 不受影响（Q_PAD/P_PAD 仅 Turing 启用，V100 字节不变；V100 已下线不可复测）

### 下一阶段大杠杆（排序，2026-09-24 中午补充逐字节预算分析）

**路线 A：BN48 + GQA x2 双头 pair + 寄存器 O**（推荐）
- 动机：pad 系列已证明 smem 流量是 sm75 prefill 主限速；双头共享一次 KV 加载
  把 KV smem 流量再砍半，是剩余唯一没动过的大流量项
- 结构：cfg<32(BLOCK_M=行=2头x16tok), BN, 256, REGS_O>；行映射
  row<16->(head0, start_row+row), row>=16->(head1, start_row+row-16)；
  grid 按 kv 头成对（gqa=6 -> 3 pair，块数/2）；mask 按 token 行、dst 按头散射；
  复用 781861f2e 已过 470/470 的 REGS 机制（warp 私有 scratch 版）
- **预算逐字节（64KB 硬顶，现行 Q8/P8/KV8 pads 计入）**：
  | 方案 | smem | 判定 |
  |---|---|---|
  | BN64 + REGS + warp 私有 8KB scratch | 71.9KB | ✗ 死 |
  | BN64 + REGS + 4KB 分相 scratch | 67.8KB | ✗ 死 |
  | BN64 + 全 pad 摘除 + 4KB scratch | 65.8KB | ✗ 差 288B，死 |
  | **BN48 + REGS + 4KB scratch（pads 保留）** | **56.4KB ✓** | **唯一健康轴** |
  | BN32 + pair | 预算松 | solo 已证 -10.6%，与 pair 增益对冲净值不明 |
- 风险：BN48 为未测新轴（QK mma 3x16 n-tile、softmax 48 列通用路径应兼容）；
  双头映射调试（mask 行=token、dst 按头散射、kv_ceiling 行循环通用）
- 预期：+5~12%（pp4096 2697 -> 2830~3020，对旧路径 3054 差距 -11.7% -> -1%~+7%）

**路线 B：旧路径数据流整体移植**（mma.sync + KQ_C 寄存器分 + Q_in_reg + ldmatrix swizzle）
- 直接复制赢家形态（旧路径 sm75 实测 3054），预期逼近 +14%
- 工作量最大：QK 循环反转（k 外层）、KQ_C 常驻、Volta/Turing mma 操作数布局
  （fragment 教训在案，动手前先单 warp probe）、seam/fixup 可不需要（单 CTA 版）
- 多轮次工程，适合作为 A 打平后的下一级

**路线 C：到此为止** —— +9.4% 已落袋（2464->2697）、全绿、已推送，写总结收工

**辅助**：FlashInfer 经典 attention 内核拉源研究（网络 + 阅读成本，可嵌入 A/B 任一线）

**决策与执行（2026-09-24 下午）**：用户选 A，已执行完毕。

### 路线 A 执行记录（GQA x2 pair + 寄存器 O，translategemma f16 r3）
| 配置 | pp1024 | pp4096 | tg32 | 判定 |
|---|---|---|---|---|
| solo 基线（Q8P8） | 2978 | 2697 | 108.0 | 对照 |
| pair BN48（无 M 门控） | 2992 | **2747 (+1.9%)** | 105.7 (-2.1%) | decode 回归 -> 加门控 |
| **pair + M>=128 门控（最终）** | 2994 (σ118 噪声) | **2732 (σ1.3)** | **108.08（平）** | 采用 |

- 门禁：470/470 x2、sweep 7747/7747（pair 版）+ 7277/7277（门控版）、Hello world. 均通过
- 累计：pp4096 基线 2464 -> 2732 = **+10.9%**；对旧路径（3054）-19.1% -> **-10.6%**
- 机制：`cfg_sm75_pair = cfg<32,48,256,true,true>`；双头 band 行映射（紧凑 c-space 经
  act_row/band_of/tok_of 展开）；REGS_O warp 私有 scratch（复用 781861f2e 已验证机制）；
  smem 61664B；派发 M>=128 走 pair、小 M（decode）回 solo 保住 tg 平价
- 教训：pair 首版未按 M 区分 -> decode 也走 pair，tg 掉 2.1%；A/B 数字以门控版为准
- 日志：`sm75-优化存档/pair-bn48-experiment.log`、`pair-mgate128.log`

**决策状态**：路线 A 完成并落库。用户选定 B（2026-09-24 下午）；B 勘察完成、实施按下述档案在后续 session 执行（多轮次工程，见档案说明）。FlashInfer 研究为辅助。

### 路线 B 设计档案（勘察完成 2026-09-24 下午；原子已全部核对，未动内核代码）

**为什么不是本 session 完成**：旧路径 2437 行、线程组织与我们根本不同（threadIdx.y 放
Q 列、np 并行 warp、KQ_max 按列 shfl），移植=把旧内核体+我方外壳重组（网格/kv_ceiling/
sinks/dst/pair band），300+ 行新代码 + 3-6 轮构建门禁，单 session 上下文会在半途留下
坏树。勘察先行=下个 session 零重复阅读直接开工。

**已核实的旧路径核心机制（全部 file:line 可回查）**：
1. KQ 寄存器累加（fattn-mma-f16.cuh:925-975）：A=K 行=KV 从 tile_K[kv][D] plain
   ldmatrix（stride=tile_stride(nbatch_K2)）；B=Q 从 tile_Q[k][ncol] k-major（stride
   DKQ/2+4）；C=tile<16,8,float>；Q_in_reg=true 时 Q_B[k_KQ_0/J] 全 D 预载 -> Q 每块只读一次
2. mask 直接入 KQ_C（:1003-1010）：get_i(l)->i(KV 行)、get_j(l)->j(Q 列)、
   tile_mask[j*(nbatch_fa+8)+i]；v1 可改直接 __ldg gmem 省掉 tile_mask 预取
3. softmax 全寄存器（:1015-1155）：KQ_max_new 按列 + shfl_xor 归约、exp 原地变 P、
   KQ_rowsum 累加；全程无 __syncthreads、无 sS/sP
4. rescale（:1158-1215）：KQ_max_scale（uint32 FTZ 技巧）乘 VKQ_C 与 KQ_rowsum
5. P->PV（:1217-1230）：B[k] = get_transposed(get_half2(KQ_C[k]))（cols_per_warp==8）；
   V 用 load_ldmatrix_trans 载为 A（:1265-1275）；VKQ_C 的 C=(DV x Q)，最终写回需转置
6. 类型原子（mma.cuh:69-360）：Turing tile<16,8> get_i=((l/2)*8)+(x/4)、
   get_j=((x%4)*2)+(l%2)；Volta C-layout 不同（32x8，get_i=(l&2)+(x&~2)）上游已双分支
   -> 同一代码未来可服务 V100（等 7.3 回线启用）

**勘察发现的硬约束**：
- nbatch_fa/BLOCK_N 必须是 32 的倍数（config static_assert）-> **pair-BN48 在 B 下失效，
  改 BN64**；smem 反而大幅释放（sS/sP/sRowMax/sRowSum/sO/expd 全部消灭）：
  BN64 kv 33792 + tile_Q ~9216 + union/sV -> ~43KB <= 64KB 预算宽松
- 线程组织冲突是移植主体：旧=列式 warp 组织；我们=行式 THREADS_PER_ROW + pair band +
  kv_ceiling + sinks/dst 行契约。对策=新引擎只换 QK/softmax/PV 体，外壳（launcher、
  grid、kv_max 侧核、sinks 插入点、dst [D][H][M] 写回、pair band<->ncols2 映射）全保留

**实施路线（下个 session 直接按此开工）**：
- 载体：cfg_sm75_pair 增 MMA 引擎（cfg flag 或独立内核函数），solo/wmma 路径不动
- Q 侧：f32->f16 转置直写 tile_Q[k][row]（替代 sQ）；A=K plain、B=Q plain（均上游原式）
- mask/sinks：mask 直接 __ldg，行索引=get_j 经 pair band 映射；sinks 在 KQ_max 段后按
  Q 行插入（对偶我们现在的 per-row sink，VKQ_C 版=乘 exp_diff 寄存器版）
- dst：VKQ_C(DV x Q) 逐 frag get_i/get_j 写回 [D][H][M]（head=band、token=Q 行映射）
- 验收：470/470 x2 + 7747 + smoke；目标 pp4096 2732 -> >=2900（旧路径 3054）、tg 保持 108
- 风险登记：get_transposed/get_half2 语义对照 :1217；KQ_max shfl 按列归属在 ncols=32
  (2x16 band) 下复核；寄存器预算重估（KQ_C+VKQ_C 驻留 ~160 f32/thread，launch_bounds(2)
  可能需降为 (1)）；mask 直读 gmem vs tile_mask 预取需实测取舍

### 路线 B probe5 实证结果（2026-09-24 下午，sm75 实测 maxerr=0.0000）

操作数方向风险清零，两条配方均**精确 0**（非简并数据、含 f16 舍入的参考）：
- **QK**：A = plain ldmatrix 自 sQ[Q行][k-fast]；B = plain ldmatrix 自 sK[KV行][k-fast]；
  k 按 half2 步进 8（=16 half）循环重载累加（probe 曾漏 k 循环 -> 部分和假错）；
  `mma(S, A, B)` -> S tile<16,16,float>，**get_i = Q 行，get_j = KV 列**
- **PV**：A = `load_ldmatrix_trans(sV)`（直接吃 gmem 布局 [KV行][d-fast]，**无需转置缓冲**）；
  B = `get_half2(S)`（分数原地打包，内核里放 exp 后的 P 即可）；
  `mma(O, A, B)` -> O tile<16,16,float>，**get_i = d，get_j = Q 行**
- 否决假设：trans-B、Kt 转置缓冲、手工 sVT（half2 对单位错：对齐 d 而非 kv）、
  双操作数 smem 往返（R2 证明 PV 免往返）
- probe 六坑（后人避雷）：half2 无 operator[]；输出 __int_as_float 双转换全打成 0；
  数据生成器简并（j*5 mod5 -> 所有 K 行相同）；**漏 k 循环**（部分和 vs 全和）；
  exp_o 参考把 score(i,j) 当 score(i,kv) 求和；打印槽位互相覆盖
- 构建行：`nvcc -arch=sm_75 -DGGML_USE_CUDA -I ggml/include -I ggml/src probe5.cu -o probe5 -lcuda -lcublas`
  （需 ggml_cuda_error/ggml_abort/ggml_cuda_get_device 三个桩）
- 状态：**QK+PV 原子已实证，内核移植（pair-mma 引擎）可按设计档案直接开工**，最后的地雷已排

### 路线 B 引擎实施设计（2026-09-24 定稿，下 session 照此机械执行）

**关键陷阱（probe blockDim=32 掩盖的）**：`tile<>::get_i/get_j` 公式硬编码
`threadIdx.x`，**只在单 warp（blockDim.x==32）下正确**；引擎 kernel 是 256 线程，
必须自写 lane 版包装（`x = threadIdx.x & 31`，公式照抄 mma.cuh tile<16,16,float>
generic 分支：`get_i = ((l/2)%2)*8 + x/4`、`get_j = (l/4)*8 + (x%4)*2 + l%2`）。
mma/ldmatrix PTX 本身按硬件 lane 工作，任意 blockDim 无碍 —— 只有展开公式要换。

**块与 warp 组织（BLOCK_M=32 行=2 头 band，BN=64）**：
- 每 n-tile（64 kv）：QK = 2 m-tile x 4 n16-sub = 8 个 mma tile，**8 warp 各领 1 个：
  m_tile = w/4, sub = w%4**（满占用）；O 寄存器 frags（每 warp 16 个 d16 tile
  = 128 f32）跨整个 n 循环常驻 —— **launch_bounds 必须 (256,1)**（(,2) 会把寄存器
  压到 128/thread 不够）
- QK：A = plain ldm 自 sQ + (m*16)*Q_STRIDE + k0h（h2 步进 8，16 步全 D）；
  B = plain ldm 自 sK + (sub*16)*KV_STRIDE + k0h；`mma(S, A, B)`，
  S 出来 (i=Q行[band 内 0..15], j=KV[sub*16..+15])
- **mask 直接进 S 寄存器**：`S.x[l] = S.x[l]*scale + mask(start_row + tok_of(m*16+get_i),
  start_col + get_j)`；**act 行无效（tok_of>=valid_tokens）必须跳过 mask 读置 NEG_INF**
  （否则 M 非 16 倍数时 gmem 越界读）；mask 的 -inf 自然处理 kv 列越界，无需特判
- **跨 warp 行归约**（行的 64 列分属 4 个 warp，shfl 过不去）：scratch_max[act_row][wi]
  4 槽（wi = w%4）：写本地 max -> __syncthreads -> 各 warp 合成 new_max（含与持久
  sRowMax 比较）-> exp_diff 就地缩放 O frags（`O.x[l] *= f[get_j]`，O 的 get_j = Q 行）
  -> exp 本 warp 16 列切片 -> 写 scratch_sum[act_row][wi] -> sync -> wi==0 的 warp
  汇总写 sRowMax/sRowSum -> sync。**每 n-tile 约 3-4 个 sync**，与现行 wmma 路径同量级
- safe_max 钳制照抄现行（全 mask 行 exp(-inf)=0 -> 输出 0）
- **PV**：`A = ldm_trans(sV + (kv0)*KV_STRIDE, KV_STRIDE)`（kv0 = sub16 无需 —— 注意
  PV 的 k 走满 64 kv：k 步进 16 kv，4 步，A base = sV + (step*16)*KV_STRIDE）；
  `B = get_half2(S)`（S 此刻已含 exp 后的 P）；`mma(O[d_tile], A, B)` 累加；
  O 输出 (i=d, j=Q 行[band 内]) —— probe R2 精确 0 验证的正是此链
- **K/V union 直接复用**：QK 用 sK、寄存器化 softmax 之后再载 V 覆盖 union —— 分数
  不落 smem，union 永不冲突（这正是 wmma 路径做不到的）
- sinks：n 循环后按 act 行（q_head_id + band）取 sink，max/sum 更新 + O frags 缩放
  （get_j 行）+ 一次 regs rescale，逻辑对偶现行
- dst 终写：`out = O.x[l] / sRowSum[act]`，act = m*16 + get_j(l)，
  地址 `(start_row+tok_of(act))*stride_D2 + (q_head_id+band_of(act))*stride_D1`
- 零可见路径、kv_ceiling、debug、launcher/派发全部沿用现行 pair 外壳
- smem 预算：sQ 16896 + sKV union 33792 + sRowMax/Sum 256 + scratch 2x512 +
  oreg 结构保留但不用（8.3KB 纯余量）≈ 60.2KB < 65536 ✓（删 oreg/expd 可再省，先不删减少 diff）
- 实施形态：cfg 加 `MMA` 布尔（cfg_sm75_pair 置真），kernel 内 QK-softmax-PV 段与终写段
  `if constexpr (CFG::MMA)` 双分支；solo/v100 编译路径字节不变
- 验收：470/470 x2 + sweep + smoke；pp4096 2732 -> >=2900（旧 3054），tg 108 不动

### 路线 B 引擎首版调试纪要（2026-09-24，未过门禁未入库——代码已还原，结论全录）

**已实证的两个硬发现：**
1. **`mma.cuh::load_ldmatrix(_trans)` 同 get_i/get_j 一样硬编码裸 `threadIdx.x`**
   （`(tid/16)*(J/2)` 列偏移）——上游块宽恒 32，我们 256 线程下**除 warp0 全部错列**。
   lane 版 wrapper（asm 逐字拷贝、 threadIdx.x&31）落地后：失败用例 ERR **0.996 -> 0.563**
   （已写入本文件工作区版本，还原前记录：`ldm16x8_lane` / `ldm16x8_lane_trans`，注意
   trans 版输出寄存器序 `{0,2,1,3}` 交换必须保留）。probe5 单 warp 测不出此 bug。
2. **probe6（跨 warp 行协议隔离器，本库已入库）四轮定界**：
   - row **max 路径 32/32 全对**（scratch_max/xors/发布结构 ✓，探针即证明）
   - **row sum 32/32 全错**，但逐 tile 发布链算术自洽（ef*old+tot 逐步可复核 ✓）
   - 槽位值两行四列完全相同被证实为 mod19 周期巧合（行位移 17 + 列位移 2 ≡ 0），非串写
   - **真凶线索：t0 的槽位含 exp=1 的命中列（1.0117 ✓），t1/t2 的槽位系统性丢失
     sn 附近的大 exp 项（sub1 仅 0.0523 ≈ 只剩 ≤sn-3 的小项）** —— 即 tiles>0 的
     exp/sum 支路把接近行最大值的贡献弄丢。下一步从 S[l] 覆写时序 / exp 与 rescale 的
     寄存器别名 / sum 初值化三处下手（t0 vs t1+ 的唯一代码差异：block_n>0 分支与
     ef*old 激活）。
- 门禁实况：首版引擎 468/470（仅两例 gqa=6 大 kv 用例挂，恰是全库仅有的 pair_mma
  覆盖用例）；MMA=0 回退 470/470 ✓；真模型冒烟空输出。失败样例参数：
  `hsk=256 nh=4 nr23={6,1} kv=4096/16384 nb=512 mask=1`（gqa=6, M=512）
- 引擎实现的关键正确部分已验证可复用：warp 映射、mask 直入寄存器、跨 warp 归约骨架、
  分相 dst 散射、launch_bounds(256,1)、ORegion 三态、cfg MMA 旗标与 SmemLayout 收缩
- probe6 的价值：纯协议、块级 (32 lanes x 8 warps)、合成分数、CPU 全对照 —— 下次
  继续调试**必须先让 probe6 转绿**再动内核（原子(probe5)与协议(probe6)双绿后合并进内核）

### 路线 B 引擎：正确性全绿、性能 -35%，默认关闭待调优（2026-09-24）

**xor 根因修复后（`x = x OP shfl(x,k)` 组合式）全量验证：**
- 门禁 470/470、全量 sweep 7747/7747、回退(MMA=0) 470/470、真模型冒烟 Hello world. 全过
- probe6 修正后 PASS（max 0 bad / sum 0 bad）——双探针（原子 probe5 + 协议 probe6）从此为引擎回归的最小前哨

**A/B（translategemma f16 r3，存档 sm75-优化存档/mma-engine-ab.log）：**
| | MMA 引擎 | wmma pair | 旧路径 |
|---|---|---|---|
| pp1024 | 2387±80 | 2973±108 | 3083 |
| pp4096 | 1770.6±0.4 | **2730.8±1.2** | 3054 |
| tg32 | 107.9 | 107.7 | 107.9 |

**结论与状态**：正确但 -35% pp，未达 >=2900 目标；按「不发回归」原则
`GGML_V100_FA_MMA` 默认改为**显式1才启用**，主干行为保持 wmma pair（2731）。
两路都进门禁（回退显式 470 过）。

**性能阶梯（pp4096，全部门禁绿）：** 首版 1771 -> swizzle 1762（证伪）-> 删冗余 sync + ef 惰性化 **1813 (+2.9%)** -> Q_in_reg 窗口化 **1841.5** -> **rescale 增长守卫 2115.6 (+14.9%，累计 +20%)**；距 wmma pair 2731 收窄到 **-22.5%**。

**rescale 守卫（本 session 最大单点收益）**：寄存器 O 有 4 份 per-sub 部分和，每份每 tile 都要 128 FMA/lane 的在线 rescale（wmma 的 smem O 只 rescale 一次且 256 线程分担 -> 我们这里结构性 x4 冗余）。但 rescale 仅在**行运行最大值增长**时才非恒等，`grew |= sn[r] > old_max` 守卫后（均匀分支，全 lane 同源推导）真实负载下绝大多数 tile 直接跳过。这类"寄存器驻留换来的集中式代价"是后续继续挖的方向。

（已归档：swizzle 1762 / sync+ef 1813 / Q窗口 1841.5；Q 窗口踩坑：内层步长必须 8 h2，写成 1 会越界读+重复累加 k，曾致 GPU fault）

**性能嫌疑清单（下 session 按序 A/B，probe5/6 保回归）：**
1. ~~ldmatrix 无 swizzle~~ **已证伪（2026-09-24 swizzle 化实测 1771 -> 1762，噪声内）**：
   swizzle 本身保留（正确、省 smem 1.5KB、对齐旧路径惯例、bytes_rc 写读同图 +
   lane 版地址），但它不是瓶颈 —— 见 sm75-优化存档/mma-swizzle.log。原嫌疑文本：
   旧路径 ldmatrix 全走 fattn-swizzle bytes_rc（原 1 号嫌疑，现降级为已做无收益） —— 旧路径 ldmatrix 全部走
   fattn-swizzle 的 bytes_rc XOR swizzle（这正是它 swz=true 的原因）；
   未 swizzle 的 ldmatrix 8x8 象限读在 Turing 上有已知 bank 冲突 —— 头号嫌疑，
   修复=Q/K/V 装载改 bytes_rc 布局（中等改动，直接对标旧路径同款）
2. **sn/ef 每 warp 重复计算**：comb 扫描+16 expf x8 warps，且 S-exp 又一遍
   —— 可合并为行主小组内一次（wmma 的 THREADS_PER_ROW 结构天然只算一份）
3. 每 tile 6 个 sync（wmma pair ~5）+ 寄存器化 O 的 128 fma rescale/tile
4. **同类「集中式寄存器代价」**：sn[16] 每 lane 全 16 行的 comb 扫描（64 smem 读 + 48 fmax），
   可改为 lane i 算一行 + shfl 广播（每 lane 只需其 6 个用到的行：fi 的 2 个 + fj 的 4 个），
   预期再省 ~150 cyc/lane/tile；grew 可顺带 shfl 归约
5. ~~Q 每 k-step 重载~~ 已做窗口化（+1.6%）；剩余：全 D 常驻 Q（16 frag=64 regs，可能 spill，
   需实测）、以及**跨 warp softmax（scratch+publish+3 barrier）vs 旧路径列内 warp 组织**
   ——这是与 3054 旧路径最大的结构差异（我们的每 tile 6->5 sync + 两次 scratch 往返，
   旧路径 np-warp 列内 shfl 全在 warp 内、无 smem scratch 无 publish），列为下一个大杠杆
5. launch_bounds(256,1) 寄存器调度（wmma 是 (256,2) 编译）；ncu 因 ERR_NVGPUCTRPERM
   不可用（需 root 改 NVreg_RestrictProfilingToAdminUsers+重启），只能靠 A/B 阶梯二分
5. launch_bounds(256,1) 的寄存器调度差异（wmma 是 (256,2) 编译）
调优后目标：先 >=2731（超 wmma），再 >=2900（原验收线），tg 保持 108。

**备选（若只要数字）**：pair 形状派发级转调上游 mma 内核（20 行，pp4096 立得 ~3054），
代价=pair wmma 引擎变成无人执行的死代码、门禁不再覆盖它 -> 不推荐，除非用户明示。

## 2070 回线验证（task-6 已实测，2026-09-24 上午补做）

- 2070 重新开机后：仓库 ff 到 `87e2348c7`，**首次构建即抓到 smem 超限真回归**：
  cfg_sm75 = 66304B > Turing 64KB 上限，`cudaFuncSetAttribute` invalid argument、
  所有 FA launch abort —— P_SUB_TILE=64 与 KV 行距偏斜叠加所致。
- 修复 `cf613e2bb`：KV_PAD 按 smem 预算分级（V100 +16 halfs 字节不变、Turing +8
  halfs 仍破 512B 对齐），cfg_sm75 = 64704B ✓。V100 路径 constexpr 不变，
  无需在已下线的 7.3 上复测。
- 修复后合同验证（RTX 2070 / sm_75）：
  - `FLASH_ATTN_EXT hsk=256` 连跑 2 轮 **470/470, 470/470** ✓
  - translategemma-4b 冒烟：`Translate... Bonjour le monde.` -> **Hello, world.** ✓
- 附带 SM75 新杠杆 A/B（translategemma f16, r3，非合同项）：

| | pp1024 | pp4096 | tg32 |
|---|---|---|---|
| 新内核 | 2882 | 2471 | **108.1** |
| 旧路径 | 3083 | 3054 | 107.9 |

  tg 由历史 96.2 追至**持平**；pp4096 差距 -27%级收窄至 **-19%**。
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
## 设计档案：publish / 跨 warp softmax 重构（待办 #2，2026-09-24）

### 现状：每 tile 5 个 barrier（`fattn-v100.cuh` 的 `mma_tile_body(block_n)` lambda）

| # | 位置 | 覆盖的 hazard |
|---|------|----------------|
| E | tile 尾部/下个 tile 的 K 写入前 | 下一 tile 的 K 存 vs 本 tile PV 读 V（**K/V 共用同一个 smem union**）|
| A | K 载入后 | 本 tile 的 K 存 vs 各 warp 的 ldmatrix 读 |
| B | `sScrMax` 写后 | N_SUBS 个 sub 的部分行最大值跨 warp 可见（comb 要读全部 4 份）|
| C | `sScrSum` 写后 | N_SUBS 个 sub 的部分行和跨 warp 可见（publish 要读全部 4 份）|
| D | V 载入后 | V 存 vs PV mma 的 ldmatrix 读 |

结构事实（决定哪些能省）：

- 一行的 64 个 KV 列被 **4 个 sub-warp 切分**，行最大值/行和**天然跨 warp** —— 这就是 B/C 存在的原因；
  旧路径（np-warp 列内组织、无 scratch 无 publish）在数据布局上根本不需要它们。
- **publish 是发散串行段**：`w_id % MMA_WARP_M == 0 && lane % 4 == 0` 选中 8 个 lane，
  串行跑 16 行（每行 8 次 smem 读 + expf + 2 次写）；其余 7 个 warp 空等至 sync D。
- **E1 不变式（必须保持）**：publish 后无 barrier，正确性依赖「下一 tile 的 sync A 之前无人读
  `sRowMax`」（读 `sRowMax` 的只有 comb 与本 publish，都在 sync A 之后）。
- 已实测：comb 扫描 shfl 化只 +2.6% => **softmax 的 ALU 量不是瓶颈**，成本在 barrier 与发散段。

### 候选（按 收益/风险 排序，均为局部改动）

1. **publish 去串行化（首选）**：把 16 行从「8 lane x 16 趟」改为 `lane < 16` 各 1 行
   （`for (int r = lane; r < 16; r += 16)`）。写目标是不同行（`sRowSum[act]`/`sRowMax[act]`），无冲突；
   行只被处理一次，语义不变。串行段 16 趟 -> 1 趟（~320 条指令 -> ~20 条）。预期 +3~6%，风险低。
2. **V 载入提前到 publish 之前（次选，近乎免费）**：现顺序为 `sync C -> publish -> V 载入 -> sync D`。
   sync C 之后所有 warp 的 K ldmatrix 读已结束，故 V 写 union 是安全的；把 V 载入移到 publish 之前，
   非 warp0 的 gmem 延迟与 warp0 的 publish 重叠（总时长 ~max(publish, Vload) 而非二者之和）。
   需先确认 `load_kv_f16_to_smem` 内部没有自己的 barrier。预期 +1~3%，风险低。
3. **K/V union 分离以便更早预取**：V(BN64 x D256 f16) 约 32KB，K 同尺寸，union 已占 smem 主体
   （总量约 50.6KB / 64KB）——分离直接爆预算（A/B/C 档案已论证 BN64 变体超 64KB）。**否决**。

### 否决/存疑

- **scratch 双缓冲（parity）**：B/C 是「跨 warp 生产-消费」，双缓冲不能删掉它们；tile 尾部的 E 也不
  由 scratch 引起（是 K/V union）。**无收益，不做**。
- **期望管理**：上述全部做完可能只有 +5~10%，**不足以填补 -20.5%**。若目标是 2900，最终仍需结构性
  改变（让一行的全部列落在同一 warp = 旧路径的列向组织，或走 split-KV + combine），那是一次重写量级，
  需单独设计档与预算。

### 验证计划（每次改动同一套）

1. `./run-2070.sh sm75-优化存档/gate-2070.sh`（470/470 x2 + sweep 7747/7747）
2. 冒烟：`GGML_V100_FA_MMA=1 llama-cli ... -no-cnv </dev/null` 必须输出 "Hello world."
3. A/B：`GGML_V100_FA_MMA=1 llama-bench -p 1024,4096 -n 32 -r 3`，对照当前 pp4096 **2171.2**、
   tg32 **108.0**（回退即止损：任何一项变差就回滚，不入库）
