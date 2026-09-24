#!/usr/bin/env bash
# Standard correctness gate for the route B engine on the 2070, run from the
# synced tree: FATTN cases on both dispatch paths plus the hsk sweep.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "--- gate default ---"
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- gate MMA=1 ---"
GGML_V100_FA_MMA=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256" 2>&1 | grep -E "tests passed|backends passed"
echo "--- sweep MMA=1 ---"
GGML_V100_FA_MMA=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -p "hsk=(64|128|192|256|512|576|640)" 2>&1 | grep -E "tests passed|backends passed"
