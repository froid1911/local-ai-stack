#!/bin/sh
# DeepSeek-V4.1-Flash-1017 (ISTA-DASLab GSQ-RCO IQ2_XS, 2.25 bpw) on Strata.
#
# Reference host: Ubuntu 26.04.1, three RTX 3090s (0,1,2). The engine binary
# lives under $HOME/.local/share/strata/; the container provides the runtime
# environment it expects. Host network keeps 127.0.0.1:18080 reachable.
#
# Prerequisites: image strata-base:24.04 exists; pack
# ~/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.25bpw exists; no other
# process holds GPUs 0,1,2.
set -eu

NAME=strata-dsv41

if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "FATAL: container $NAME already exists; stop/remove it first." >&2
  exit 1
fi

exec docker run -d --name "$NAME" \
  --network host \
  --gpus all \
  -v "$HOME:$HOME" \
  -e HOME="$HOME" \
  -e STRATA_RUNTIME_DIR=/tmp/strata-rt \
  -e XDG_RUNTIME_DIR=/tmp/strata-xdg \
  strata-base:24.04 \
  bash -lc 'mkdir -p /tmp/strata-rt /tmp/strata-xdg && chmod 700 /tmp/strata-rt /tmp/strata-xdg; "$HOME/.local/share/strata/terminal/current/bin/strata" launch --model ist-das-2.25bpw --family deepseek41 --gpu 0,1,2 --context 256k --vision off --tune quick --access local --port 18080 --yes --no-tui --restart; exec sleep infinity'
