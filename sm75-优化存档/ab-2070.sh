#!/usr/bin/env bash
# Build + gate + smoke + A/B bench for the FA paths on the 2070.
# Default dispatch on sm_75 is the upstream mma kernel; the fork kernel and the
# route B engine are opt-in, so their coverage needs the env vars explicitly.
set -u
cd "$(dirname "$0")/.." || exit 1
cmake --build build -j 8 > /tmp/build_ab.log 2>&1 || { echo BUILD_FAIL; tail -25 /tmp/build_ab.log; exit 1; }
echo "BUILD_OK errors=$(grep -c ' error' /tmp/build_ab.log)"
bash "sm75-优化存档/gate-2070.sh"
M=~/models/translategemma-4b-it.i1-Q4_K_M.gguf
echo "--- smoke engine (FA=1 MMA=1) ---"
GGML_V100_FA=1 GGML_V100_FA_MMA=1 timeout 300 ./build/bin/llama-cli -m $M -p "Translate to English: Bonjour le monde." -n 40 --no-jinja -ngl 99 -no-cnv </dev/null 2>&1 | grep -A2 "Bonjour" | tail -2
echo "--- bench default (upstream mma on sm_75) ---"
./build/bin/llama-bench -m $M -p 1024,4096,8192 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -6
echo "--- bench fork FA (GGML_V100_FA=1) ---"
GGML_V100_FA=1 ./build/bin/llama-bench -m $M -p 1024,4096,8192 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -6
echo AB_DONE
