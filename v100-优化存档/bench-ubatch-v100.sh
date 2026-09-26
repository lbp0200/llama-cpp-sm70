#!/bin/bash
# V100 (SM70) llama.cpp ubatch / MMQ bench.
# Runs on the 7.3 box (Tesla V100-PCIE-32GB). Model: Qwen3.8-27B IQ4_XS.
#
#   bash bench-ubatch-v100.sh [prefill|decode|nsys|all]
#
# All runs use GGML_V100_FA=0 (upstream FA kernel = the V100 deployment choice).
set -u

REPO=${REPO:-$HOME/llama-cpp-sm70}
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf}
BENCH=$REPO/build/bin/llama-bench

export GGML_V100_FA=0
unset  GGML_CUDA_FORCE_MMQ

# llama-bench prints "| model | size | ... | test | t/s |"; field NF-1 is t/s.
ts() { awk -F'|' '{print $(NF-1)}'; }

prefill() {
    echo "=== prefill: -b/-ub pairs, r=3 ==="
    for p in 2048 16384; do
        for ub in 512 1024 2048 4096; do
            printf "pp%-6s ub=%-5s -> " "$p" "$ub"
            "$BENCH" -m "$MODEL" -p "$p" -n 0 -r 3 -ngl 99 -b "$ub" -ub "$ub" 2>&1 \
                | grep -E "^. qwen35.*pp$p" | ts
        done
    done
    echo "=== prefill: pp65536, r=2 ==="
    for ub in 512 2048; do
        printf "pp65536 ub=%-5s -> " "$ub"
        "$BENCH" -m "$MODEL" -p 65536 -n 0 -r 2 -ngl 99 -b "$ub" -ub "$ub" 2>&1 \
            | grep -E "^. qwen35.*pp65536" | ts
    done
}

decode() {
    echo "=== decode: tg128, r=3 ==="
    for ub in 512 2048; do
        printf "tg128   ub=%-5s -> " "$ub"
        "$BENCH" -m "$MODEL" -p 0 -n 128 -r 3 -ngl 99 -b "$ub" -ub "$ub" 2>&1 \
            | grep -E "^. qwen35.*tg128" | ts
    done
}

# Kernel-level attribution. Note: pp65536 gives a complete capture, decode does not
# (default trace buffer drops most of the many small decode kernels).
nsys_run() {
    echo "=== nsys kernel breakdown, pp65536 ==="
    nsys profile -o /tmp/lcb64 --force-overwrite=true --stats=true \
        "$BENCH" -m "$MODEL" -p 65536 -n 0 -r 1 -ngl 99 2>&1 \
        | sed -n '/cuda_gpu_kern_sum/,/cuda_gpu_mem_time_sum/p'
}

case "${1:-all}" in
    prefill) prefill ;;
    decode)  decode ;;
    nsys)    nsys_run ;;
    all)     prefill; decode ;;
    *) echo "usage: $0 [prefill|decode|nsys|all]" >&2; exit 2 ;;
esac
