# Architecture

## Hardware

Three RTX 3090s at indices 0, 1 and 2, 24 GB each, on a 48-thread / 247 GB
Ubuntu 26.04.1 host. There is no NVLink; inter-GPU traffic goes over PCIe.

A fourth 3090 (serial `1324221009624`) was removed in 2026-10 because its VRAM
held fixed bad cells. The fault was storage, not compute: matmuls were clean,
but index tensors could land out of range and crash the engine, or in range and
produce silent garbage. Without ECC the failure was silent. Removing the card
is why the current layouts use three GPUs.

## GPU allocation

| Workload | GPUs | Why |
|---|---|---|
| DeepSeek-V4.1-Flash IQ2_XS (Strata) | 0,1,2 | expert arena split across all three |
| Qwen3.8-27B W4A16 (vLLM) | 0,1 | TP=2; 27B INT4 on two cards keeps 262K context |
| LFM3-1.7B (llama.cpp) | 0 | 1.8 GB weights + 1.6 GB q8_0 KV at 262K, fits one card |
| Qwen3.8-Flash-Next IQ3_S / IQ4_XS (Strata) | 0,1,2 | 125B MoE experts resident across three cards |

A card holds one model at a time. vLLM servers on disjoint pairs can coexist,
but the Strata container and the 27B vLLM server overlap on GPUs 0 and 1, so
they are mutually exclusive.

## Why tensor parallelism matters

Single-stream decode is bandwidth-bound: every step reads weights and KV from
the cards. Only tensor parallelism divides bytes-per-GPU-per-step. For 27B
INT4 (17.3 GiB weights), TP=2 on two 3090s is the validated layout: weights
split roughly in half, KV fits at 262K, and measured decode is 57–73 tok/s.
TP=4 was measured only while the faulty card was in the loop and is not a safe
configuration.

## Container boundary

The DeepSeek engine runs in `strata-base:24.04` with host networking and
`$HOME` bind-mounted. The original reason was a glibc floor; the container
also isolates the engine's runtime directories (`STRATA_RUNTIME_DIR`,
`XDG_RUNTIME_DIR`) from the host. The host's three GPUs are passed through
with `--gpus all`.

## Agent layer

OMP is the harness; its `models.yml` declares the local OpenAI-compatible
endpoints and `config.yml` maps model roles onto them. Herdr owns the
terminals: agent sessions live in panes, and Herdr reports working / blocked /
idle state. This is the part that makes a local stack usable without a cloud
control plane.
