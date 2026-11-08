# OMP / oh-my-pi

OMP is the coding-agent harness. Its config lives under `~/.omp/agent/`.

- Upstream: https://github.com/can1357/oh-my-pi · https://omp.sh
- Version used here: `omp/18.18.1`

## Install

Follow the upstream install instructions, then copy these files:

```bash
cp config.yml   ~/.omp/agent/config.yml
cp models.yml   ~/.omp/agent/models.yml
cp AGENTS.md    ~/.omp/agent/AGENTS.md
cp env.example  ~/.omp/agent/.env
```

`AGENTS.md` is the personal workflow note: work on `fix/*` or `feat/*`
branches, small logical commits, and Powerhouse worktrees via
`scripts/new-worktree.sh`.

## Providers in `models.yml`

All local providers use `auth: none` and an OpenAI-compatible endpoint:

| Provider | Endpoint | Model | Notes |
|---|---|---|---|
| `strata-dsv41` | `127.0.0.1:18080/v1` | `deepseek-v4.1-flash` | 256K context, reasoning in `reasoning_content` |
| `strata` | `127.0.0.1:8090/v1` | Qwen3.8-Flash-Next IQ3_S / IQ4_XS | 262K context declared manually |
| `vllm` | `127.0.0.1:8003/v1` | Qwen3.8-27B W4A16 | multimodal checkpoint |
| `vllm-defect` | `127.0.0.1:8003/v1` | Qwen3.8-27B uncensored W4A16 | historical faulty-card slot |
| `lfm3` | `127.0.0.1:8310/v1` | LFM3-1.7B | llama.cpp, native tool calls |
| `tess` | `127.0.0.1:18090/v1` | Tess-4.5-27B | historical two-card layout |
| `openrouter` | cloud | `deepseek/deepseek-v4.1-flash-1017` | optional fallback |

The `vllm` and `vllm-defect` entries both point at `:8003` in the snapshot;
check what is actually listening before relying on either. The `tess` entry
was built for GPUs 2+3, which is not the current three-card layout.

## Model roles in `config.yml`

Every role used for real work maps to `strata-dsv41/deepseek-v4.1-flash`:

| Role | Assignment |
|---|---|
| `slow`, `commit`, `worker`, `smol` | `strata-dsv41/deepseek-v4.1-flash:auto` |
| `plan`, `task`, `default` | `strata-dsv41/...` then OpenRouter fallback |
| `review` | `strata-dsv41/deepseek-v4.1-flash` |
| `vision` | `strata/qwen3.8-flash-next-iq3_s:auto` |
| `tiny` | `local/lfm2.5-350m` then `strata-dsv41` |

The `tiny` role references a `local` provider that is not defined in
`models.yml`; add that provider or change the role if you copy this config.
Claude-specific roles were removed from this snapshot — the stack is local.

`maxInFlightRequests.vllm: 8`, `streamFirstEventTimeoutSeconds: 600`, and
`update.auto: false` are the other non-default choices.
