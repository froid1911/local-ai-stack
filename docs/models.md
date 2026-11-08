# Models

All local endpoints are OpenAI-compatible. Reasoning models return
chain-of-thought in `reasoning_content`; llama.cpp and vLLM use their native
tool-call parsers.

## Production

### DeepSeek-V4.1-Flash-1017 — Strata, port 18080

- Pack: ISTA-DASLab GSQ-RCO IQ2_XS, 2.25 bpw
- Engine: Strata 0.2.12, engine 0.1.58
- GPUs 0,1,2; 256K context; vision off
- Measured 2026-11-08: 62–66 tok/s decode with DSpark drafter on;
  40–44 tok/s with drafter off. Prefill ~530 tok/s on a fresh 6.6K prompt.
- OMP model id: `strata-dsv41/deepseek-v4.1-flash`

### Qwen3.8-Flash-Next — Strata, port 8090

- 125B MoE, two packs: IQ3_S (`~/Strata-data/packs/iq3_s`) and
  UD-IQ4_XS (`~/Strata-data/packs/unsloth-ud-iq4_xs`)
- 262K context; experts resident across GPUs 0,1,2
- Measured: ~140 tok/s decode (IQ4_XS, 2026-10-07); 112–137 tok/s (IQ3_S,
  2026-10-08); prefill 4,455 tok/s at 75K for IQ3_S
- Only one variant is served at a time; the JSON config selects the pack
- Strata reports its context as `meta.n_ctx`, which OMP's discovery does not
  read, so `models.yml` declares `contextWindow: 262144` explicitly

### Qwen3.8-27B W4A16 AutoRound — vLLM, port 8003

- dbirks AutoRound, INT4 group-128 symmetric; 19 GB checkpoint
- TP=2 on GPUs 0,1; 262K context; fp8 KV; prefix caching on
- Measured on the healthy pair (2026-09-06): 72.9 tok/s shallow,
  57.4 tok/s at 179K; 6/6 needle retrievals at 62K and 169K
- Multimodal checkpoint: the vision tower is kept out of the int4 quantisation;
  `models.yml` declares image input manually

### LFM3-1.7B Q8_0 — llama.cpp, port 8310

- LiquidAI LFM3-1.7B, 1.8 GB Q8_0 GGUF; 262K native context
- One RTX 3090; q8_0 KV is ~1.6 GB at full context
- Measured 2026-10-28: 310 tok/s decode, ~20K tok/s prefill, 142K-token needle
  retrieved verbatim
- Chat template is ChatML + `think`; llama-server's Qwen3-Coder detector turns
  its XML tool calls into native OpenAI `tool_calls`
- Thinking is on/off via `chat_template_kwargs: {enable_thinking: false}`

### Tess-4.5-27B — llama.cpp, port 18090 (historical layout)

- Qwen3.8-27B finetune, Q4_K_M + MTP draft; 262K context; vision projector
- Measured 57–65 tok/s decode (thinking on, MTP)
- Built for GPUs 2+3; the current host has only 0,1,2, so this layout is not
  part of the production run

## Not included

Model weights and Strata packs are not in the repo. Checkpoints are 19 GB for
the 27B, 1.8 GB for LFM3, and multi-tens-of-GB for the Strata packs. The
launch scripts document the expected paths and download/build sources.
