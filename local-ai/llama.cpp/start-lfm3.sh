#!/usr/bin/env bash
# LiquidAI LFM3-1.7B, Q8_0 GGUF, via llama.cpp. Port 8310, one RTX 3090.
#
# arch `lfm3` (llama.cpp src/models/lfm3.cpp + llama-memory-hybrid-iswa.h):
#   24 layers = 13 gated short-conv (head_count_kv=0 -> recurrent, no KV cache),
#   5 sliding-window attention (window 256, rope base 1e4), 6 global attention
#   (rope base 1e6). n_embd 2048, 16 q heads / 4 kv heads, head dim 128,
#   ffn 7680, vocab 128k, native ctx 262144, shortconv cache 7.
#   Only the 11 attention layers hold KV: 6 global layers dominate
#   (4 kv heads x 128 dim x 2 tensors x 2 B = 2 KB/token/layer -> 12 KB/token
#   f16, ~6 KB/token q8_0 -> 1.6 GB at 262144; the SWA layers cache ~256 tok).
#   Weights Q8_0 = 1.85 GB, so the whole model is ~4 GB of 24 GB even at full ctx.
#
# Task: local agent / tool-calling / structured-output model, 17 languages.
# Sampling from the checkpoint metadata: temp 0.1, top_k 50, repeat_penalty 1.1.
# MEASURED 2026-10-28 on GPU 0 (Q8_0, q8_0 KV, one 3090):
#   pp512 19.7K tok/s, pp4096 20.2K tok/s, tg128 310 tok/s;
#   needle at 141,965 prompt tokens retrieved verbatim in 16 s.
# Chat template (embedded in the GGUF) is ChatML +  thinking; tool calls are
# <tool_call><function=NAME><parameter=NAME>VALUE</parameter ...> which
# llama-server's Qwen3-Coder detector parses into native OpenAI tool_calls.
# Use the plain OpenAI endpoint (no in-band dialect needed). Thinking can be
# switched off per request with chat_template_kwargs: {enable_thinking: false}.
set -euo pipefail

BASE=~/models/LFM3-1.7B-GGUF
MODEL="$BASE/LFM3-1.7B-Q8_0.gguf"
BIN=~/llama-cpp-cuda/build/bin/llama-server   # master build b10297+ (has src/models/lfm3.cpp)

PORT=${PORT:-8310}
CTX=${CTX:-262144}      # native, no YaRN needed
KVTYPE=${KVTYPE:-q8_0}  # 1.6 GB at full ctx; f16 would be 3.2 GB -- both fit

[ -f "$MODEL" ] || { echo "Model missing: $MODEL"; exit 1; }
[ -x "$BIN" ]   || { echo "llama-server missing (rebuild llama-cpp-cuda): $BIN"; exit 1; }

export CUDA_VISIBLE_DEVICES=${GPUS:-0}
export LLAMA_ARG_HOST=127.0.0.1

exec "$BIN" \
  --model "$MODEL" \
  --alias lfm3-1.7b \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k "$KVTYPE" --cache-type-v "$KVTYPE" \
  --parallel 1 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.1 \
  --jinja \
  "$@"
