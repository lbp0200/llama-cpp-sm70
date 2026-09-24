#!/usr/bin/env bash
# Build + gate + smoke + A/B bench for the route B engine on the 2070.
# Run from the Mac: ./run-2070.sh sm75-优化存档/ab-2070.sh
set -u
cd "$(dirname "$0")/.." || exit 1
cmake --build build -j 8 > /tmp/build_ab.log 2>&1 || { echo BUILD_FAIL; tail -25 /tmp/build_ab.log; exit 1; }
echo "BUILD_OK errors=$(grep -c ' error' /tmp/build_ab.log)"
bash "sm75-优化存档/gate-2070.sh"
M=~/models/translategemma-4b-it.i1-Q4_K_M.gguf
echo "--- smoke MMA=1 ---"
GGML_V100_FA_MMA=1 timeout 300 ./build/bin/llama-cli -m $M -p "Translate to English: Bonjour le monde." -n 40 --no-jinja -ngl 99 -no-cnv </dev/null 2>&1 | grep -A2 "Bonjour" | tail -2
echo "--- bench MMA=1 ---"
GGML_V100_FA_MMA=1 ./build/bin/llama-bench -m $M -p 1024,4096 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -5
echo "--- bench default ---"
./build/bin/llama-bench -m $M -p 1024,4096 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16 2>&1 | tail -5
echo AB_DONE
