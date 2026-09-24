#!/usr/bin/env bash
# Upstream FA path (GGML_V100_FA=0) vs the fork's default FA on the 2070,
# across prefill lengths, plus decode. Deployment decision data.
# Run from the Mac: ./run-2070.sh sm75-优化存档/upstream-vs-fork.sh
set -u
cd "$(dirname "$0")/.." || exit 1
M=~/models/translategemma-4b-it.i1-Q4_K_M.gguf
echo "--- fork FA (default, route A pair) ---"
./build/bin/llama-bench -m $M -p 1024,4096,8192,16384 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -10
echo "--- upstream FA (GGML_V100_FA=0) ---"
GGML_V100_FA=0 ./build/bin/llama-bench -m $M -p 1024,4096,8192,16384 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -10
echo UP_DONE
