# Instructions for llama.cpp (TurboQuant fork)

## Project Overview

This repo is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream) that adds the TurboQuant feature set on top of a fully-synced upstream base. The local tree always contains all of upstream master plus fork additions; rebasing onto latest upstream is a recurring task.

### What TurboQuant adds

TurboQuant compresses the KV cache far beyond the standard `q8_0` by applying a fixed 128x128 orthonormal Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`) to cache vectors before quantization, which Gaussianizes the distribution, and inverse-rotating after dequantization. Head dims that are not multiples of 128 are zero-padded. MLA models (DeepSeek) have no separate V cache, so V rotation/padding is skipped for them, and K/V cache types must be identical.

The five fork-only GGML types (registered in `ggml/include/ggml.h`):

| Type   | Enum                       | Purpose                             | Size         |
|--------|----------------------------|-------------------------------------|--------------|
| turbo2 | `GGML_TYPE_TURBO2_0` (43)  | KV cache only                       | 2 bits/value |
| turbo3 | `GGML_TYPE_TURBO3_0` (44)  | KV cache only                       | 3.25 bits    |
| turbo4 | `GGML_TYPE_TURBO4_0` (47)  | KV cache only                       | 4.25 bits    |
| TQ3_1S | `GGML_TYPE_TQ3_1S` (45)    | model weights, WHT-rotated Lloyd-Max| 3 bits, block 32 |
| TQ4_1S | `GGML_TYPE_TQ4_1S` (46)    | model weights                       | 4 bits, block 32 |

Turbo cache types are runtime-only, never stored in GGUF. TQ3_1S/TQ4_1S are first-class weight types with CPU, CUDA/HIP (warp-cooperative mmvq), Metal, Vulkan, and SYCL kernels, exposed as `llama-quantize` targets.

### Key files

- `ggml/src/ggml-turbo-quant.c` - the codec (keep byte-identical to fork tip)
- `ggml/include/ggml.h` - type enum 43-47, `GGML_OP_TURBO_WHT`
- `src/llama-kv-cache.cpp` - cache wiring, `get_k_idx`, layer-adaptive precision
- `src/llama-graph.cpp` - inverse-WHT post-processing (FA and non-FA paths)
- `ggml/src/ggml-cuda/mmvq-tq.cu` - native TQ dp4a kernels (`GGML_TQ_NATIVE=1`)
- `ggml/src/ggml-vulkan/` - turbo FA, SET_ROWS, dequant shaders
- `ggml/src/ggml-metal/ggml-metal.metal` - TurboFlash kernels
- `docs/KV-cache-quantization.md` - authoritative usage doc (read before touching cache types)

### Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`/`q8_0`/`turbo2`/`turbo3`/`turbo4` for K and V is supported. Turbo cache types require flash attention; it is auto-enabled with a warning. Quantized V with FA explicitly disabled is an error (upstream behavior). The same flags work in `llama-server`, `llama-bench`, `llama-perplexity`.

### SM75 dev/test target (RTX 2070, 10.1.2.16)

- The 10.1.2.16 box (RTX 2070 / sm_75, 8 GB) is the SM75 test GPU, but as of
  2026-09-26 it is **back in production** and no longer a free box: `llama-server`
  runs on :8080 (translategemma-4b + a COK LoRA) and :8081 (base model), plus the
  `llama-gateway` front end, together holding ~6 GB of the 8 GB. Check before
  touching it:
  ```bash
  ssh bolt-remote   # ssh alias (HostName 10.1.2.16)
  systemctl list-units --state=running | grep -E 'llama|gateway'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  ```
  **Do not disable or kill those services** (an earlier revision of this file told
  you to - that is stale). GPU correctness runs there are usually still fine - a
  full `test-backend-ops` gate passed alongside them on 2026-09-26 without a
  restart - but do not treat a number measured there as a benchmark, and do not
  run anything that needs more than the ~2 GB left free.
- **Final target model on the 2070: `translategemma-4b-it.i1-Q4_K_M.gguf`**
  (~2.5 GB, `~/models/` on the box, also mirrored under `~/Models/` on the dev
  Mac). It is the reference for all SM75 A/B: head_dim 256, GQA 2, f16 KV
  friendly.
- Build on the box: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`.
- **Code sync: `./sync-2070.sh`** (rsync over the `bolt-remote` alias; copies the whole
  tree including uncommitted work, excludes `build/`, `.pi/` and the probe binaries, then
  re-pins the box's `origin` to an anonymous read-only https URL). The Mac is the only
  pusher, so **the 2070 needs no GitHub credentials and no private key** - do not register
  a key for it or run `git push` there. A stale `~/.ssh/id_rsa` copy may exist on the box;
  nothing in this workflow uses it.
- **Both sm_70 and sm_75 dispatch to upstream's `fattn-mma-f16.cuh`.** The fork Volta/Turing
  FA kernel (`fattn-v100.cuh`) was deleted on 2026-09-26 after it measured slower on both
  architectures back-to-back (V100 pp2048 -3.4% / pp16384 -20.9% / pp65536 -37.1%; 2070 -8%
  at pp4096 to -15% at pp16384). Do not resurrect it without a same-card upstream baseline
  first - that mistake cost a whole campaign.
- **Run repo scripts on the box with `./run-2070.sh <script-in-repo>`** - it syncs, then runs the
  script there with stdin closed (`llama-cli` otherwise eats the remaining script lines and the log
  silently ends early). `./run-2070.sh sm75-优化存档/gate-2070.sh` is the standard FATTN gate.
- **Before any GPU work on either box, check what is already using it.** The 2070
  is in production and the V100 may be borrowed; `nvidia-smi --query-compute-apps`
  and `systemctl list-units --state=running` take two seconds and have already
  caught one mistake (a full gate run launched at a production box).
- **The 2070's serving config is measured, not guessed.** `--cache-type-k/v q8_0`
  costs ~18% decode (f16 KV is ~4 MB at `-c 1024`, so the compression buys nothing),
  and the runtime `--lora` costs ~28% prefill / ~23% decode because `build_lora_mm`
  adds 476 tiny matmuls per step (`llama-graph.cpp:1534`). **Merging the LoRA is a
  wash - do not retry it**: Q4 loses the delta to quantization error, Q6_K keeps it
  but its 28% extra bytes cancel the saved kernels. Full evidence, including three
  claims this investigation retracted: `sm75-优化存档/2070-tg-lora-调查-2026-09-26.md`.
  Note the deployed binary is NOT this tree's build (`/usr/local/bin/llama-server`,
  version 10830 / `a3d5603d1`, a commit not present here), so per-request numbers
  from it are not comparable to `build/bin/llama-server`.
- The V100 box (192.168.7.3, sm_70, 32 GB) stays the large-model/long-context
  validation environment (Qwen3.8-27B IQ4_XS etc.). `ssh -i ~/.ssh/id_rsa lbp@192.168.7.3`.
  It was clean (no services, 0 MiB used) as of 2026-09-26.
- **On the V100, `-b 2048 -ub 2048 --spec-type draft-mtp` is the deployment config.**
  As of 2026-09-26 a **Volta-only build (`-DCMAKE_CUDA_ARCHITECTURES=70`) already
  defaults `n_ubatch` to 2048**, so on the V100 no flag is needed at all; `-ub`
  still overrides. The gate is build-time (`LLAMA_VOLTA_ONLY_BUILD` in the root
  CMakeLists.txt) because ggml exposes no cc accessor, so an sm_75 build is
  untouched and the 2070 keeps 512. `--spec-type draft-mtp` adds +44% decode on real
  prose, for 552 MiB and no second model file. Measured on the V100: prefill +20~37%
  and decode 37.0 -> 53.4 tok/s, generated text byte-identical, +378 MiB VRAM. Root
  cause of the prefill win: Volta is the only NVIDIA arch whose prefill goes through
  "dequantize the weights to f16, then cuBLAS" (`should_use_mmq` falls through to the
  `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` threshold in `ggml-cuda/mmq.cu`), so the dequant
  cost is `ceil(prompt/ub) x weight bytes` and grows with context. Evidence, raw logs
  and a rerun script: `v100-优化存档/`.
- **MTP speculation needs `--spec-type draft-mtp`; `-md` alone does nothing.** In this
  version `-md <draft>` only sets the draft path, while `common_params_speculative::types`
  stays `{NONE}`. Without `--spec-type` the server logs `[spec] loading draft model`, reserves
  the VRAM, and then runs plain decoding - it looks like "speculation did not help" when it
  never ran. Also `--draft-max`/`--draft-min` do not exist here; the flags are
  `--spec-draft-n-max` / `--spec-draft-n-min`. Do not raise the depth above the default 3:
  MTP acceptance decays fast with depth (54.8% at 3, 23.8% at 8) and deeper drafting loses.
- **The V100's absolute tok/s drifts up to ~10% between jobs** (clock falls from 1530 MHz to
  1260-1275 MHz under sustained load). Only within-job comparisons are trustworthy - every
  headline number in `v100-优化存档/` comes from a single job containing both sides.
- The FA campaign (`feature/v100-fa-port`) is closed; its code was deleted on 2026-09-26, leaving
  two genuine bug fixes behind (`fattn-vec.cuh`'s D512 VEC stub, `ggml-cuda.cu`'s graph null-deref
  guard). The reasoning, evidence and negative results are in `Volta-Turing-FA进度.md` and
  `v100-优化存档/`.
- **SM75 gotcha (cost a full debug session): the 1Cat WMMA fragment ->
  (row, col) expansion only holds for Volta (HMMA.884). Turing (HMMA.16816)
  has a different fragment element order, so mask/scale applied through the
  fragment lands on wrong positions. The kernel now applies scale + mask on
  the smem tile by explicit (row, col) after `store_matrix_sync`, which is
  architecture independent. Never trust a fixed fragment expansion across
  archs; verify with a single-warp probe (`probe4.cu`) if in doubt.

### SM75 optimization reference (vLLM-2080Ti-Definitive)

Study the Turing optimization reference at `/Users/lbp/SharedData/vLLM-2080Ti-Definitive`
(a dual-RTX-2080Ti / SM75 vLLM fork by `weicj`) before making SM75-specific
kernel or runtime decisions. Its validated SM75 stack, in rough value order:

- **FlashInfer / FlashQLA-SM70-SM75** (`weicj/FlashQLA-SM70-SM75`): the prefill
  attention route for Qwen on Turing; see also its `csrc/attention/` headers.
- **Marlin** quantized GEMM (W4A16/W8A16) - the model-weight side of SM75.
- **TurboQuant / INT8-FP8 KV cache** - compressing KV on a bandwidth-starved card.
- **CUDA Graph (FULL/PIECEWISE + AOT)** - Turing graphs are valuable, never
  disable them to "fix" a bug; fix the underlying bug instead.
- **MTP / DFlash2 speculative decoding** - decode token/s on sm_75 is low
  (~30-40 tok/s raw), speculation is how Turing serving stays responsive.

Evidence from that repo (Qwen3.8-27B on 2x2080Ti 22GB, TP2, FP16 KV, no
speculation): prefill 4K token ~1694 tok/s, decode ~33 tok/s; with MTP/4 or
DFlash2 decode jumps to ~95-220 tok/s. 145-512K contexts validated. So SM75 can
serve large models fast at prefill while decode needs speculation.

Applicability boundary for our 2070 (8 GB, single card, no NVLink):
- TP2/multi-card profiles DO NOT apply.
- Weights above ~7 GB do not fit; the 2070 target stays `translategemma-4b`
  (D=256, GQA 2) unless we go smaller/denser.
- The attention-kernel and CUDA-graph lessons DO apply to our fork
  (`feature/v100-fa-port` Volta FA dataflow, low-smem Turing variant,
  graph null-deref fix).

Fork/TurboQuant overlaps: the repo's TQ4NC KV cache ciphertext matches our
fork's turbo KV family in spirit (both bound KV bytes on Turing-class cards).

### Environment knobs

| Variable                    | Default | Effect |
|-----------------------------|---------|--------|
| `TURBO_LAYER_ADAPTIVE`      | `0`     | Layer-adaptive KV precision; `7` = Boundary V (edge layers q8_0, middle turbo) |
| `TURBO_AUTO_ASYMMETRIC`     | `1`     | Auto-select asymmetric K/V types for large-GQA models |
| `TURBO_SPARSE_V`            | `1`     | Sparse-V dequant skip in flash attention |
| `GGML_TQ_NATIVE`            | unset    | `1` opts out of load-time TQ->q8_0 conversion, uses fused native TQ kernels (saves ~1.7x VRAM on decode-heavy workloads) |
| `GGML_CUDA_FUSE_CHAIN`      | unset    | `0` disables the elementwise chain fusion (SILU/GELU/ADD/MUL/SCALE/CLAMP runs into one kernel, `ggml_cuda_fuse_elem_chain`) |
| `GGML_CUDA_Q8CACHE`         | unset    | `0` disables the per-graph shared-quantize cache in mmvq (gate and up projections reuse one q8_1 copy of the activation) |
| `LLAMA_ATTN_ROT_K/V_OVERRIDE` | off   | Optional upstream attention rotation (TurboQuant manages its own rotation) |
| `GGML_CUDA_FORCE_MMQ`       | unset    | Route prefill GEMM through fused mmq instead of dequant+cuBLAS. On the V100 it is only worth it together with a large `-ub`: +0.5~1.3% at `-ub 2048`, but **-5.2% at `-ub 512`**. Do not enable globally |
| `GGML_TQ_MMQ`               | unset    | Native TQ prefill mmq path (`ggml-cuda.cu:1968`). Same mechanism as `GGML_CUDA_FORCE_MMQ`, for TQ3_1S/TQ4_1S weights. Untested |

Note: `-ub` is a CLI flag, not an env var, but it belongs in this table's spirit - see
`v100-优化存档/README.md` for the V100 numbers.

### Test gates (all must pass before touching quant/backend code)

- `test-turbo-quant` - turbo3 basis MSE=0/Cosine=1.0, turbo4 Cosine=0.9956
- `test-quantize-fns` - includes TQ3_1S/TQ4_1S and rotated-domain buffer sizing
- `test-backend-ops` - full sweep on CPU + CUDA0 (23k+ cases on the RTX 5090 dev box); rejects 0/0 as FAIL
- `llama-bench` with `-ctk/-ctv turboN`; type parser accepts `tq3_1s`/`tq4_1s`

### What the test gates do and do not cover

What each suite does:

- `test-turbo-quant` - CPU codec round-trip quality: quantize -> dequantize -> CPU inverse WHT, MSE/cosine on fixed vectors, plus a chunked-dequant invariance check for all five turbo types at row lengths straddling the vec_dot chunk size. No GGML graphs, no backend kernels.
- `test-quantize-fns` - CPU quantize/dequantize functions against error budgets, including TQ3_1S/TQ4_1S. Skips TURBO2_0/3_0/4_0 by design: their dequant output stays in the WHT-rotated domain.
- `test-backend-ops` - per-op GGML graphs, run on each backend and compared numerically against the CPU reference. This is the only gate that exercises backend kernels.
- `llama-bench` - tokens/s on real models. Timing only; it never checks output correctness.

Coverage limits (each caused a real miss):

- `test-backend-ops` used to report `Backend ...: OK` when every case was skipped because the backend verdict was `n_ok == tests_run`, and 0/0 passed. It now fails the backend when no test ran. Issue #242 remains open for reporting which graph node caused a case to be unsupported.
- The generic SET_ROWS sweep has a view variant with `r/2` rows. At r=1 that is 0 rows: the case writes nothing and passes for every type in `all_types`, including TQ4_1S.
- The MUL_MAT_ID sweep used n=16 only, and the mat-vec decode path is selected only when `src2->ne[1] <= 8` (`ggml_vk_use_mul_mat_vec_id`). n=16 exercises mul_mm_id only; MoE decode was never touched. The n=1 cases and the DSv4-shaped sweep (commit 637300387, PR #269) now cover both sides of that threshold.
- The harness initializer wrote quantized tensors with one packed `ggml_backend_tensor_set`, which copies `size` bytes contiguously and never strides by `nb[1]`. For a strided view (the `k_v > k` MUL_MAT cases view `k` rows of a `k_v`-row base) the data landed at `i*row_size` instead of `i*nb[1]` and the last rows were never written; the CPU reference read the stale tail and produced NaN, which presented as the CUDA backend failing because CPU is the reference and is skipped as a backend under test. Fixed by row-by-row init for non-contiguous tensors (issue #268, PR #276). The TQ4_1S `k_v=1600` case now passes: the CUDA NaN #276 observed no longer occurs because PR #277 gates the fused TQ mul_mat paths on contiguous `src1`/`dst`, routing this view to the stride-aware fallback.

A green run means the cases that ran passed, not that your change was exercised. Check that your cases actually ran:

- `-o` filters on the op name from `ggml_op_desc` (e.g. SET_ROWS). The dedicated turbo write tests have their own names (SET_ROWS_TURBO3, SET_ROWS_TURBO4, SET_ROWS_TQ4_1S); filter with those, or they never run.
- Watch for `not supported [backend]` lines and `0/0 tests passed`.

### Git workflow

- Remotes: `origin` = `lbp0200/llama-cpp-sm70` (this repo, the only remote configured). There is no
  `upstream` remote; `ggml-org/llama.cpp` master reaches us inside `origin/feature/turboquant-kv-cache`,
  which tracks the TurboQuant fork (itself a fork of ggml-org master). Add
  `upstream` = `https://github.com/ggml-org/llama.cpp` explicitly if you ever need to compare against
  raw ggml-org master.
- Branches: `feature/turboquant-kv-cache` = the upstream TurboQuant fork; work happens on a branch on
  top of it (currently `feature/v100-fa-port`, which is where the V100 work and the two fork bug fixes
  live).
- Syncing upstream is a plain merge: `git fetch origin && git merge origin/feature/turboquant-kv-cache`.
  Upstream master is always fully contained in the tree, and `git log` is the record of the last sync
  point. Merges have been conflict-free; the one repeated overlap to expect is `fattn-vec.cuh` /
  `fattn.cu` / `template-instances/fattn-vec-instance-*.cu`, where upstream and this fork have twice
  fixed the same D512 VEC ptxas smem overflow. Upstream's `GGML_USE_HIP` guard won; do not re-add a
  fork-side stub there.
- After any sync, rebuild and gate on both boxes: `./sync-73.sh --rebuild` plus
  `v100-优化存档/gate-v100.sh` for sm_70, and `./run-2070.sh sm75-优化存档/gate-2070.sh` for sm_75.

### Known pitfalls (each caused a real bug once - check these first on regressions)

- **Metal**: turbo kernels need their `[[host_name]]` instantiations; a missing one is a NULL-pipeline deref on the first turbo KV write.
- **Vulkan**: SET_ROWS pipeline registration must include TURBO2_0/3_0/4_0 with `require_full_subgroups=true, subgroup_size=32`, or every turbo KV write aborts.
- **CUDA dispatch**: TQ weights must be excluded from the mmvq path before the fused-TQ branch (`ggml_cuda_should_use_mmvq`), or `GGML_TQ_NATIVE=1` aborts.
- **CUDA TQ4_1S decode**: the centroid LUT in `mmvq-tq.cu` decodes through `get_int_from_table_16`, then re-interleaves even/odd bytes with constant selectors (`__byte_perm(v.x, v.y, 0x5140 / 0x7362)` on nvcc and MUSA, `__builtin_amdgcn_perm` on HIP), the same pattern `vecdotq.cuh` already uses. The old garbage output (NMSE ~1.0) came from the earlier permute chain, not from constant selectors. Verified numerically on GB10 (sm_121) and MI210 (gfx90a). Gate any change to this function on a CUDA-side `test-backend-ops -o MUL_MAT -p type_a=tq4_1s` run on an NVIDIA card plus the AMD run, not on a clean compile.
- **DeepSeek/MLA**: K and V cache types must be identical; turbo FA auto-enable runs before upstream's quantized-V FA check.
- **MoE models**: the small-batch TQ `MUL_MAT_ID` path routes experts on device and stays CUDA-graph capturable (`[TAG_MUL_MAT_ID_CUDA_GRAPHS]` in `ggml-cuda.cu`); buffers it hands to kernels must outlive every captured graph, so caches retire outgrown buffers instead of freeing them. The large-batch path dequantizes to f16 cuBLAS and synchronizes the stream.
- **gguf-py**: keep model constants deduplicated; stacked-duplicate merge artifacts crash `import gguf`.

> [!IMPORTANT]
>
> AI-assisted development is encouraged in this fork. Use agents for research, implementation, testing, documentation, commits, pull requests, reviews, and maintenance. Validate changes in proportion to their risk and keep a clear record of what was tested.

---

## Guidelines for Contributors

A PR represents a long-term commitment - maintainers must review, integrate, and support the code indefinitely. What matters is whether the change is correct, understandable, tested, and maintainable.

A working, in-scope PR is **not** enough on its own to get merged. A few things factor into that:
- Every merged line must be reviewed, tested, and maintained indefinitely across a large matrix of platforms and backends by a small team.
- llama.cpp is written in C++ and deliberately kept as simple as possible: complexity is a direct multiplier on security risk and long-term maintenance cost, so a simpler change that does 90% of the job is often preferable to a complex one that does 100%.
- What matters most is technical understanding, evidence, and willingness to maintain the change long-term.
- Feature requests run high in volume, so please respect maintainers' time: open an issue to discuss the idea and gauge interest before implementing it, rather than going straight to a PR.

Contributors must:
1. **Understand their code fully** - use AI assistance freely, but verify important claims and design choices.
2. **Own maintenance** - address bugs and respond thoughtfully to feedback.
3. **Communicate directly** - be concise, specific, productive, and positive without being sappy.
4. **Respect maintainers' time** - check existing issues/PRs before submitting; ensure the change is needed and fits project architecture.

### AI-Assisted Development

AI assistance is welcome throughout the development workflow, including:

- Learning, exploration, debugging, and codebase research
- Design analysis, implementation, refactoring, and mechanical work
- Tests, benchmarks, documentation, and release notes
- Commit messages, PR descriptions, issue reports, and reviewer responses
- Code review, review comments, CI investigation, and maintenance
- Commits, pushes, branch management, PR operations, and comments for any contributor when requested or included in the assigned workflow

This policy applies equally to every contributor and agent working in this repository. Agents may complete an assigned workflow end to end. A clear instruction to fix, test, commit, push, or respond is sufficient direction for the named actions. Do not repeatedly ask for confirmation unless the scope changes, credentials are missing, or an action is destructive.

AI attribution is optional. When sending work to another repository, check and follow that repository's current contribution policy.

---

## Guidelines for AI Coding Agents

Every PR requiring review consumes finite maintainer capacity. Before assisting with any submission, verify:
- The proposed changes and their tradeoffs are understood
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions

Agents should inspect relevant code before editing, make reasonable assumptions when safe, test in proportion to risk, and clearly report uncertainty or incomplete coverage.

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing code, read the relevant files and understand the existing patterns. Changes must blend in with the surrounding codebase. For a large change or new pattern, explain the approach and tradeoffs before implementation. Ask for direction only when the scope or design requires a meaningful user choice.

Common mistakes to avoid:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.

### Code Comment Examples

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

Commit message:

```
// GOOD: Write a concise commit

llama : fix KV being cleared during context shift


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: gather context, then complete the authorized workflow
gh search issues
gh search prs
rg ...
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentations:
- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
