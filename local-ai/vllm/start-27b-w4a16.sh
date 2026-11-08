#!/usr/bin/env bash
# UPDATE 2026-10-30: the faulty card (serial 1324221009624, stuck-at-1 bits) is REMOVED.
# flank now has 3 GPUs: 0,1 as before, and GPU 2 = the healthy former GPU 3 (serial
# 1322221042697). TP=4 configs cannot run until a 4th card is back; a 27B TP=2 pair
# is only GPUs 0,1 (GPU 2 alone has 24GB). The GPU-2 warnings below are historical.
# ============================================================================
# Qwen3.8-27B W4A16 (dbirks AutoRound, INT4 group-128 symmetric) -- vLLM 0.28, TP=4, port 8003.
# Arch qwen3_5 / Qwen3_5ForConditionalGeneration: 64 layers, hybrid GDN
# (16 full-attention + 48 linear-attention), head_dim 256, native 262K ctx.
# ============================================================================
# STATUS: **VALIDATED ON GPUs 0,1 (TP=2) -- 2026-09-06.** GPU 2 HAS FAULTY VRAM; it is excluded
# by default. Do NOT set GPUS=0,1,2,3 until that card is replaced. See "THE REAL BUG" below.
#
#   MEASURED on the HEALTHY pair (spec=none, kv=fp8, pc=off, ctx=262144, TP=2, GPUS=0,1;
#   streaming, so TTFT is excluded from the decode figure):
#     shallow (214 tok)       decode  72.9 tok/s
#     deep (179,415 tok)      decode  57.4 tok/s   TTFT 174.8s (1027 tok/s prefill)
#     correctness  6/6 needles retrieved -- 3/3 at 62,334 tok AND 3/3 at 169,022 tok
#     concurrency  PASS: 141,630-tok prefill under an active 4,000-tok generation, needle
#                  retrieved, no assert, no engine death. This is the case that killed
#                  EVERY TP=4 config.
#   Still beats the FP8 PROD server (69 tok/s shallow) on HALF the GPUs.
#   NOTE prefill is FASTER at TP=2 than TP=4 (1027 vs 846 tok/s): prefill is compute-bound and
#   TP=4's extra PCIe all-reduce costs more than the added compute wins. No NVLink on this rig.
#
#   Earlier TP=4 figures (95.6 shallow / 81.0 deep, 2,006,704 KV tokens) were measured WITH the
#   faulty card in the loop. The speeds are real but the config is not safe to run.
#
# ---------------------------------------------------------------------------
# WHY TP=4 AND NOT 1 GPU / 4 REPLICAS  (asked 2026-09-06, settled by arithmetic)
# ---------------------------------------------------------------------------
# Single-stream decode is bandwidth-bound: every step reads weights + KV off the card.
# What matters is BYTES PER GPU PER STEP, and only tensor parallelism divides them.
# INT4 weights are 17.3 GiB; a 3090 does 936 GB/s.
#   layout            weights/GPU   roofline    realistic    max ctx
#   1 GPU, no TP        17.3 GiB      ~50/s       ~40/s       ~55K   <- 2.5x SLOWER, ctx collapses
#   2 replicas x TP=2    8.65 GiB     ~100/s      ~50/s      ~200K
#   TP=4 (this)          4.33 GiB     ~200/s   ~100-110/s    262K   <- best on BOTH axes
#   4 replicas x 1 GPU  17.3 GiB   ~50/s each  ~40/s each    ~55K each
# 4 replicas wins only on AGGREGATE throughput for >=4 concurrent users (~160 tok/s summed),
# at worse per-request latency and 1/5 the context. It is a throughput play, not a latency one.
# On one card, 17.3 GiB of weights leaves <5 GiB for KV on a 24 GiB 3090 => ~55K tokens.
# Same lesson as llama.cpp `--split-mode layer`, which ran the GPUs SEQUENTIALLY and capped
# at ~30 tok/s. vLLM TP=4 aggregates bandwidth; that is the whole reason this config is fast.
#
# ---------------------------------------------------------------------------
# THE REAL BUG (2026-09-06): GPU 2 HAS FAILING VRAM. IT IS NOT A vLLM PROBLEM.
# ---------------------------------------------------------------------------
# Pattern test (write `arange ^ 0x5A5A5A5A` to ~20 GiB, sync, read back, compare):
#     GPU0 0 mismatches | GPU1 0 | **GPU2 4437 (1479 EVERY pass)** | GPU3 0
# Fixed bad cells (~5.9 KB), and COMPUTE is fine on all four (bitwise-identical matmuls) --
# it is storage, which is why it corrupts *index* tensors. These 3090s have NO ECC: silent.
# One fault, two faces:  index lands OUT of range -> `ScatterGatherKernel.cu:203 index out of
# bounds` -> EngineCore dead.  Index lands IN range -> silent '!!!!' garbage, no crash.
# `dmesg | grep Xid` showed 5/5 events on PCI 0000:4d:00 = GPU 2, including one during STARTUP
# memory profiling -- no long context, no concurrency. That is what ruled out every software
# theory (fp8 KV, cudagraph mode, max-model-len, chunked prefill, INT4 itself): all were tested
# and all were wrong.
# **RULE: when failures cluster on ONE TP rank across UNRELATED code paths, suspect the card.**
# With TP=N, rank K is always the same physical GPU, so bad hardware perfectly mimics a
# rank-specific software bug.
# **FIX: replace GPU 2.** Until then GPUS=0,1 TP=2. TP must divide the 4 KV heads => 1, 2 or 4
# ONLY, so "just use the other three cards" is impossible.
#
# RE-TESTED 2026-09-12 (bench/memtest-gpu.py, same arange^0x5A5A5A5A method):
#   GPU2 1465 stable bad int64 words on ALL 3 passes (first at byte ~3.5GB)
#   GPU3 0 mismatches.  (09-06: 4437/1479 int32 words -- same cluster.)
# New face of the same fault (09-12): on a TP=2 (GPUS=2,3) layout BOTH the
# uncensored FP8 and the official Qwen3.8-27B-FP8 produce deterministic
# token-soup -- on vLLM 0.23 AND 0.28 (venv-independent), MTP on AND off --
# while W4A16 AutoRound INT4 on the same pair is clean (output identical to
# the healthy 0,1 pair). Severity is LAYOUT-DEPENDENT: the bad cells sit at
# fixed physical addresses and whatever tensor each quantization's allocation
# places there decides the outcome -- FP8's layout lands them where the model
# collapses, INT4's where it tolerates. Slot status: DEAD for FP8, USABLE
# for the W4A16/INT4 family until the card is replaced (verify with a deep
# needle, bench/slot8003-check.py -- 'so far clean' is not 'safe').
# 09-12 RESULT: first verification run on the slot PASSED -- noon-at-cgn
# Uncensored-W4A16-AutoRound (raw BF16 embed/lm_head, same g128-sym recipe):
# 262K ctx, 697,517 KV tokens, shallow 66 tok/s clean, 188,902-tok needle
# retrieved, deep decode 55 tok/s clean. Now serves 8003 as
# qwen3.8-27b-uncensored-w4a16.
#
# ---------------------------------------------------------------------------
# THE OLD INT4 WARNING -- NOW BELIEVED TO BE THE HARDWARE FAULT ABOVE
# ---------------------------------------------------------------------------
# A DIFFERENT INT4 checkpoint (RedHatAI/Qwen3.8-27B-INT4) FAILED output validation at
# 188,862 tokens in ALL THREE spec configs -- see bench/RESULTS.txt:
#     int4-none      FAIL-VALIDATION   needle missed, output '!!!!!!!!'
#     int4-mtp       FAIL-VALIDATION   needle missed, output '!!!!!!!!'
#     int4-dflash-4  FAIL-VALIDATION   needle missed, FLUENT prose  <- the dangerous mode
#     fp8-*          PASS (all three)
# That matrix ran with prefix caching FORCED OFF, so INT4's collapse at depth is NOT merely a
# prefix-caching artifact -- the earlier "prefix caching destroys INT4" note is too narrow.
# That matrix ran TP=4, i.e. WITH the faulty GPU 2 in the loop, so it is not evidence about
# quantisation at all. Re-run it on GPUs 0,1 before believing any INT4-vs-FP8 quality claim.
# This checkpoint (AutoRound, group-128 symmetric, MTP head + linear_attn in_proj + lm_head left
# unquantized) is 6/6 clean on healthy silicon at 62K and 169K.
# => PC still defaults to 0: prefix caching is a SEPARATE, still-untested risk on INT4 (it forces
#    Mamba 'align' mode). Validated on FP8 only. Earn it back with a needle test at depth.
# => A BROKEN CONFIG GENERATES GARBAGE FAST. Never quote tok/s without asserting on output
#    text at the SAME depth. Harness: bench/matrix-w4a16.py.
#
# ---------------------------------------------------------------------------
# KNOBS  (all env vars; defaults are the safe/fast pair, not the fastest-looking pair)
# ---------------------------------------------------------------------------
#   CTX=262144    max-model-len. Native window; do not exceed without YaRN + revalidation.
#   SEQS=16       max-num-seqs. Single-stream latency barely cares; raise for concurrency.
#   GPUS=0,1      which physical GPUs. GPU 2 IS FAULTY -- keep it out (see below).
#   TP=2          tensor-parallel size. Must divide 4 KV heads => 1, 2 or 4 only. NOT 3.
#   SPEC=none     none | mtp | dflash. See below -- none is expected fastest at depth.
#   NSPEC=4       draft length for dflash. 2/4/6 are within noise on FP8; 8 COLLAPSES.
#   KV=fp8        auto | fp8 | fp8_per_token_head | int8_per_token_head.  <-- THE LEVER
#   PC=1          prefix caching. VALIDATED on INT4 (healthy GPUs) 2026-09-07: cold 77.4s ->
#                 warm 0.6-0.7s at 94K tokens = 129x, and warm reads at 50%/92% depth
#                 answered DIFFERENT questions correctly, so restored GDN state is sound.
#                 The old PC=0 default came from the GPU-2 corruption, not from caching.
#   MNBT=         max-num-batched-tokens; empty = vLLM default (8192). Try 16384 for prefill.
#   ASYNC=0       async scheduling. Matched baseline within noise on FP8; kept as a knob.
#   CGMODE=       cudagraph mode. Empty = vLLM default FULL_AND_PIECEWISE. Set PIECEWISE to
#                 drop FULL decode graphs -- a suspect in the concurrency crash (see CRASH below).
#   EAGER=0       --enforce-eager. Diagnostic only; costs a lot of decode speed.
#   PORT=8003     8003 so this does NOT collide with the FP8 PROD server on 8002.
#   NAME=qwen3.8-27b-w4a16   served-model-name. Set to qwen3.8-27b-fp8 for pi drop-in.
#
# SPEC: speculation is a TRAP at depth on this arch. The drafter attends over the whole KV
#   every step, so its cost scales with context while acceptance falls (0.90 -> 0.84).
#   Measured on FP8, decode tok/s: 1K 88(mtp) vs 69(none); 32K 27 vs 65; 64K 15 vs 61.
#   MTP wins only below ~4K. DFlash2 has 5 layers over the full context -- worse at depth.
#
# KV: --kv-cache-dtype fp8 halves KV traffic (64 KB -> 32 KB/token). At 262K that takes
#   bytes/GPU/step from ~8.5 to ~6.4 GiB, projecting ~80 tok/s instead of ~60, and it doubles
#   the KV token budget. This was previously blocked by the nvcc-10.1 FlashInfer JIT wall --
#   that wall is GONE, fixed by the CUDA_HOME + NVCC_APPEND_FLAGS pair set below.
#   fp8 KV runs at scale 1.0 here (kv_cache_scheme=null, and vLLM 0.28 dropped
#   --calculate-kv-scales). That is normally a corruption risk -- but it was VALIDATED at 179K:
#   needle retrieved, output clean. It is also what buys the 2M-token KV budget.
#
# ---------------------------------------------------------------------------
# CLOSING THE LAST 20% (81 -> 100 tok/s at full context) -- UNTESTED IDEAS, IN ORDER
# ---------------------------------------------------------------------------
#  1. KV=int8_per_token_head or turboquant_4bit_nc. At 179K the KV term is roughly half of
#     bytes/GPU/step; taking KV from 8 to 4 bits should give ~+15%. Per-token-head scaling is
#     the accuracy-preserving variant. Highest expected value, and the harness already gates it.
#  2. MNBT=16384. Prefill-side only; will not move decode. Worth it for the 212s TTFT, not tok/s.
#  3. SPEC=mtp ONLY if your workload lives under ~4K. It should reach ~120 tok/s there and it
#     collapses to single digits at depth. Never set this on a long-context server.
#  4. NOT worth trying: --enforce-eager (slower), 1-GPU or 4-replica layouts (see table above),
#     async scheduling (matched baseline within noise on FP8).
#
# CTX ceiling: prefill is ~880 tok/s and FLAT with depth, so a cold 262K prompt is ~5 min to
#   first token and no flag fixes that (GA102 roofline puts the TTFT floor near 165s at 250K).
#   If you need long context to feel fast, the answer is prefix REUSE, not decode tuning.
# ============================================================================
set -euo pipefail

# MODEL: swap checkpoints without a second script. Both are int4 compressed-tensors.
#   .../Qwen3.8-27B-W4A16-AutoRound  (dbirks AutoRound, default)
#   .../Qwen3.8-27B-INT4             (RedHatAI, llm-compressor AWQ+GPTQ)
MODEL=${MODEL:-/home/froid/models/Qwen3.8-27B-W4A16-AutoRound}
VENV="/home/froid/vllm-env-028"

CTX=${CTX:-262144}
SEQS=${SEQS:-16}
SPEC=${SPEC:-none}
NSPEC=${NSPEC:-4}
KV=${KV:-fp8}      # VALIDATED default; see measurements above
PC=${PC:-1}   # VALIDATED on INT4 2026-09-07: 129x warm TTFT, warm reads correct
MNBT=${MNBT:-}
ASYNC=${ASYNC:-0}
CGMODE=${CGMODE:-}   # empty=vLLM default (FULL_AND_PIECEWISE) | PIECEWISE | FULL_DECODE_ONLY | NONE
EAGER=${EAGER:-0}
# GPUS/TP: GPU 2 (PCI 4d:00) has FAULTY VRAM -- 1479 fixed bad int32 cells, confirmed
# 2026-09-06 (GPUs 0/1/3 clean). It corrupts index tensors -> device-side asserts and
# '!!!!' garbage. EXCLUDE IT. TP must divide the 4 KV heads, so TP=3 is impossible:
# use TP=2 on two healthy cards until the card is replaced.
GPUS=${GPUS:-0,1}
TP=${TP:-2}
PORT=${PORT:-8003}
# NAME: serve as `qwen3.8-27b-fp8` to be a DROP-IN for pi's `local-27b-fp8` provider without
# editing ~/.pi/agent/models.json. Honest default is the real name -- aliasing a 4-bit model
# under an "fp8" id is a footgun if you later A/B them.
NAME=${NAME:-qwen3.8-27b-w4a16}
DRAFT="/home/froid/models/Qwen3.8-27B-DFlash2"

# ---- preflight ------------------------------------------------------------
[ -f "$MODEL/config.json" ] || { echo "FATAL: model missing/incomplete: $MODEL" >&2; exit 1; }
[ -d "$VENV" ]              || { echo "FATAL: venv missing: $VENV" >&2; exit 1; }

# GPUs hold ONE model at a time. Fail loudly rather than OOM 4 minutes into loading.
# Only the GPUs THIS instance will use -- two servers on disjoint pairs is a
# supported layout, so a busy GPU 0 must not block a run on GPUs 2,3.
held=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
if [ -n "$held" ]; then
  echo "FATAL: GPUs are held by PID(s): $(echo "$held" | tr '\n' ' ')" >&2
  echo "       Run ~/models/stop-inference.sh first (it kills by PID from nvidia-smi;" >&2
  echo "       do NOT pkill -f 'vllm serve' -- that pattern matches the calling shell)." >&2
  exit 1
fi
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FATAL: port $PORT already in use." >&2; exit 1
fi

source "$VENV/bin/activate"
# nvcc-13 for FlashInfer JIT (system nvcc is 10.1 and cannot build sm_86 kernels against CUDA 13).
# Without these two, fp8 KV dies with "nvcc fatal: Unknown option '-generate-dependencies-with-compile'".
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_APPEND_FLAGS='-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'   # venv nvcc 13.2 vs headers 13.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# vLLM logs "Reducing Torch threads from 24 to 1 for serving. Set OMP_NUM_THREADS in the
# external environment" -- it clamps to 1 itself, but each TP worker would otherwise start
# with a 24-thread OpenMP pool before the clamp, oversubscribing this 24-core CPU by TP x 24.
# Serving is GPU-bound; the CPU side is launch + sampling glue. Set it explicitly.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES="$GPUS"

# ---- assemble -------------------------------------------------------------
EXTRA=()
case "$SPEC" in
  none) ;;
  mtp)    EXTRA+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}') ;;
  dflash) [ -f "$DRAFT/config.json" ] || { echo "FATAL: DFlash2 drafter missing: $DRAFT" >&2; exit 1; }
          EXTRA+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NSPEC}") ;;
  *) echo "FATAL: SPEC must be none|mtp|dflash (got '$SPEC')" >&2; exit 1 ;;
esac

# vLLM 0.28 turns prefix caching ON BY DEFAULT -- omitting the flag is NOT enough.
# The only proof it is off is the `enable_prefix_caching=False` line in the startup log.
if [ "$PC" = "1" ]; then EXTRA+=(--enable-prefix-caching); else EXTRA+=(--no-enable-prefix-caching); fi
[ "$KV" != "auto" ] && EXTRA+=(--kv-cache-dtype "$KV")
[ -n "$MNBT" ]      && EXTRA+=(--max-num-batched-tokens "$MNBT")
[ "$ASYNC" = "1" ]  && EXTRA+=(--async-scheduling)
[ -n "$CGMODE" ]    && EXTRA+=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\"}")
[ "$EAGER" = "1" ]  && EXTRA+=(--enforce-eager)

echo "=== Qwen3.8-27B W4A16 | ctx=$CTX seqs=$SEQS spec=$SPEC kv=$KV pc=$PC cg=${CGMODE:-default} tp=$TP gpus=$GPUS port=$PORT ==="
echo "=== verify in the log: enable_prefix_caching=$([ "$PC" = 1 ] && echo True || echo False), the quant kernel, GPU KV cache size ==="

exec vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$CTX" \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs "$SEQS" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  "${EXTRA[@]+"${EXTRA[@]}"}"
