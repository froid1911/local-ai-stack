#!/usr/bin/env bash
# Stop whatever holds the GPUs -- by PID from nvidia-smi, NOT by matching command lines.
# History: pattern matching bit me four times. `pkill -f 'vllm serve'`, `pgrep -f 'bin/vllm'`,
# and even `pgrep -f "[v]llm serve"` inside this script all matched the CALLING shell (whose
# command line happened to contain the pattern, e.g. from a heredoc), killing the caller.
# nvidia-smi gives the actual compute-app PIDs, which is unambiguous.
set -u
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
if [ -n "$pids" ]; then
  echo "stopping GPU holders: $pids"
  for p in $pids; do kill "$p" 2>/dev/null; done
  for i in $(seq 1 30); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && break
    sleep 2
  done
  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u)
  for p in $left; do kill -9 "$p" 2>/dev/null; done
  sleep 3
fi
# vLLM's API-server parent can outlive the workers and hold the port; it has no GPU context.
for p in $(pgrep -x vllm 2>/dev/null); do kill "$p" 2>/dev/null; done
sleep 1
echo "GPUs now:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
