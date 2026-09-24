#!/usr/bin/env bash
# Run a script from this repo on the 2070 test box.
# The tree is synced with rsync first, so no separate copy step is needed:
#   ./run-2070.sh sm75-优化存档/gate-2070.sh
# stdin is closed because llama-cli would otherwise consume the rest of a
# script stream and silently drop every command after it.
set -euo pipefail
HOST=${HOST:-bolt-remote}
DEST=${DEST:-llama-cpp-sm70}
[ $# -ge 1 ] || { echo "usage: $0 <script-in-repo> [args...]"; exit 1; }
script=$1; shift
./sync-2070.sh >/dev/null
ssh "$HOST" "cd $DEST && bash $script $(printf '%q ' "$@") </dev/null"
