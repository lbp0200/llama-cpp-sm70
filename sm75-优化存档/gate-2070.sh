#!/usr/bin/env bash
# Standard correctness gate for the FA kernels on the 2070, run from the
# synced tree. On sm_75 the default dispatch is the upstream mma kernel, so
# the fork kernel and the route B engine are covered by explicit env runs.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "--- default dispatch (upstream mma on sm_75) ---"
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- fork FA (GGML_V100_FA=1) ---"
GGML_V100_FA=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- fork FA + route B engine ---"
GGML_V100_FA=1 GGML_V100_FA_MMA=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- sweep (fork FA) ---"
GGML_V100_FA=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=(64|128|192|256|512|576|640)" 2>&1 | grep -E "tests passed|backends passed"
echo "--- turbo KV smoke (default dispatch) ---"
M=~/models/translategemma-4b-it.i1-Q4_K_M.gguf
timeout 300 ./build/bin/llama-cli -m $M -p "Translate to English: Bonjour le monde." -n 40 --no-jinja -ngl 99 -no-cnv -ctk turbo3 -ctv turbo3 </dev/null 2>&1 | grep -A2 "Bonjour" | tail -2
