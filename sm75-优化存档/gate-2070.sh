#!/usr/bin/env bash
# Standard correctness gate for the FA paths on the 2070, run from the synced
# tree. Both sm_70 and sm_75 now dispatch to upstream's fattn-mma-f16.cuh by
# default (the fork FA kernel was deleted), so this covers the default path plus
# the turbo KV path.
set -u
cd "$(dirname "$0")/.." || exit 1
B=./build/bin/test-backend-ops

echo "--- default dispatch (upstream mma) ---"
$B -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- sweep ---"
$B -o FLASH_ATTN_EXT -p "hsk=(64|128|192|256|512|576|640)" 2>&1 | grep -E "tests passed|backends passed"
echo "--- turbo KV smoke: the decode path must run (Generation line) ---"
# llama-cli's conversation banner eats the completion text, so assert on the
# generation timing line instead: it only prints if the turbo KV decode ran.
# Text-level turbo KV verification is in v100-优化存档/README.md (llama-server).
timeout 600 ./build/bin/llama-cli -m $M -p "The capital of France is" -n 16 -ngl 99 -no-cnv \
    -ctk turbo3 -ctv turbo3 </dev/null 2>&1 | grep -E "Generation:|error|failed" | tail -2
