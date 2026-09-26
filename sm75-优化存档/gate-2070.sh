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
M=${M:-$HOME/models/translategemma-4b-it.i1-Q4_K_M.gguf}
# llama-cli's conversation banner eats the completion text, so assert on the
# generation timing line instead: it only prints if the turbo KV decode ran.
# Text-level turbo KV verification is in v100-优化存档/README.md (llama-server).
# This box is in production (~6 GB of 8 GB held by llama-server), so a model load
# can legitimately fail on VRAM - report that clearly instead of a bare error.
out=$(timeout 600 ./build/bin/llama-cli -m "$M" -p "The capital of France is" -n 16 -ngl 99 -no-cnv \
    -ctk turbo3 -ctv turbo3 </dev/null 2>&1)
if echo "$out" | grep -q "Generation:"; then
    echo "$out" | grep -E "Generation:" | tail -1
elif echo "$out" | grep -qiE "failed to load|out of memory"; then
    echo "SKIPPED: could not load the model - check nvidia-smi; this box runs production"
else
    echo "FAILED: no Generation line"
    echo "$out" | tail -5
fi
