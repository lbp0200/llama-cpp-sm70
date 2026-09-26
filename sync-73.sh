#!/usr/bin/env bash
# Sync this working tree to the V100 box (192.168.7.3) over rsync.
# The V100 box is a build/bench box: it receives the tree as-is (uncommitted work
# included) and keeps its own build/ directory. It needs no GitHub credentials
# and no private key for this workflow. Same contract as sync-2070.sh.
#
#   ./sync-73.sh              # sync + report the box's HEAD
#   ./sync-73.sh --rebuild    # sync, then rebuild on the box (needed after any
#                             # .cu/.cuh change: the box's build/ is not synced)
set -euo pipefail
HOST=${HOST:-lbp@192.168.7.3}
KEY=${KEY:-$HOME/.ssh/id_rsa}
DEST=${DEST:-llama-cpp-sm70}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -i "$KEY")

rsync -az --delete --human-readable \
  --exclude 'build/' \
  --exclude '.pi/' \
  --exclude '__pycache__/' \
  --exclude 'probe4' --exclude 'probe5' --exclude 'probe6' \
  -e "ssh ${SSH_OPTS[*]}" \
  ./ "$HOST:$DEST/"

ssh "${SSH_OPTS[@]}" "$HOST" "cd $DEST && git remote set-url origin https://github.com/lbp0200/llama-cpp-sm70.git && git log --oneline -1 && git status -sb | head -4"

if [ "${1:-}" = "--rebuild" ]; then
  echo "--- rebuilding on the box"
  ssh "${SSH_OPTS[@]}" "$HOST" "cd $DEST && cmake --build build -j\"\$(nproc)\" 2>&1 | tail -15"
fi
