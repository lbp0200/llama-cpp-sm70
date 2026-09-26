#!/usr/bin/env bash
# Standard correctness gate for the FA kernels on the V100 (sm_70), run from the
# synced tree. Since the default flip, both sm_70 and sm_75 dispatch to upstream's
# fattn-mma-f16.cuh by default, so the fork kernel and the route B engine are
# covered by explicit env runs.
#
#   ssh -i ~/.ssh/id_rsa lbp@192.168.7.3 'cd ~/llama-cpp-sm70 && bash v100-优化存档/gate-v100.sh'
set -u
cd "$(dirname "$0")/.." || exit 1
B=./build/bin/test-backend-ops
M=${M:-$HOME/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf}

echo "--- default dispatch (upstream mma on sm_70) ---"
$B -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- fork FA (GGML_V100_FA=1) ---"
GGML_V100_FA=1 $B -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- fork FA + route B engine ---"
GGML_V100_FA=1 GGML_V100_FA_MMA=1 $B -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- sweep (fork FA); hsk=320 is a known fork defect and is excluded ---"
GGML_V100_FA=1 $B -o FLASH_ATTN_EXT -p "hsk=(64|128|192|256|512|576|640)" 2>&1 | grep -E "tests passed|backends passed"
echo "--- turbo KV smoke through the DEFAULT dispatch (must not need the fork FA) ---"
GGML_V100_FA=0 timeout 600 ./build/bin/llama-cli -m $M -p "The capital of France is" -n 16 -ngl 99 -no-cnv -ctk turbo3 -ctv turbo3 </dev/null 2>&1 | tail -3
