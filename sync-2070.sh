#!/usr/bin/env bash
# Sync this working tree to the 2070 test box (bolt-remote) over rsync.
# The 2070 is a build/bench box: it receives the tree as-is (uncommitted work
# included) and keeps its own build/ directory. It needs no GitHub
# credentials and no private key for this workflow.
#
#   ./sync-2070.sh              # sync + report the box's HEAD
#   ./sync-2070.sh --rebuild    # sync, then configure (if build/ is missing) and
#                               # build. Needed after any .cu/.cuh change, since
#                               # build/ is not synced.
set -euo pipefail
HOST=${HOST:-bolt-remote}
DEST=${DEST:-llama-cpp-sm70}
rsync -az --delete --human-readable \
  --exclude 'build/' \
  --exclude '.pi/' \
  --exclude '__pycache__/' \
  --exclude 'probe5' --exclude 'probe6' \
  -e ssh \
  ./ "$HOST:$DEST/"
# keep the box on an anonymous read-only origin (the Mac is the only pusher)
ssh "$HOST" "cd $DEST && git remote set-url origin https://github.com/lbp0200/llama-cpp-sm70.git && git log --oneline -1 && git status -sb | head -4"

if [ "${1:-}" = "--rebuild" ]; then
  echo "--- configure + build on the box"
  # nvcc is only on PATH in a login shell, so use bash -lc.
  ssh "$HOST" "cd $DEST && bash -lc 'set -e; [ -d build ] || cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > /tmp/cfg_sm75.log 2>&1; cmake --build build -j8' 2>&1 | tail -8"
fi
