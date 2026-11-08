# Local AI stack repository — design

## Intent

Create a public GitHub repository under `froid1911` that documents and
preserves the configuration of this workstation's local AI stack, so the setup
can be shared on Twitter and reproduced by someone with similar hardware.

## Goals

- One repository containing the real configs for Herdr, OMP, and the local
  inference servers.
- A README that explains the hardware, model lineup, ports, and measured
  performance in plain language.
- Production launch scripts only: DeepSeek V4.1 container, Strata IQ3_S/IQ4_XS,
  vLLM 27B W4A16, llama.cpp LFM3, and the GPU-stop script.
- No credentials, tokens, session databases, logs, or model weights.

## Non-goals

- Reproducing upstream source (Herdr, OMP, Strata) — links only.
- Including experimental launch variants from `~/models`.
- Including Claude configuration or Claude-specific model roles.

## Constraints

- Public repository, MIT license for this repo's contents.
- Configs contain host-specific absolute paths; README must say so.
- The reference host has three GPUs, not four; historical comments in scripts
  are preserved, and GPU lists were updated to `0,1,2` where they mattered.
- The `vllm` and `vllm-defect` entries in `models.yml` both point at `:8003`;
  the README documents that as a snapshot caveat rather than silently fixing it.
- `omp/config.yml` contains a `tiny` role pointing at a `local` provider that
  is not defined in `models.yml`; documented as a caveat.

## Approaches considered

1. **GitHub profile repo `froid1911/froid1911`** — README renders on the
   profile; best for a single Twitter landing page. Rejected: mixes personal
   profile content with a config dump.
2. **Dedicated public repo `froid1911/local-ai-stack`** — chosen. Clean link
   target, configs live in subfolders, README is the stack writeup.
3. **Private repo** — rejected: the point is sharing the setup publicly.

## Decisions

- Repo: public `froid1911/local-ai-stack`.
- Structure: `herdr/`, `omp/`, `local-ai/`, `docs/`.
- Production scripts only; experimental variants excluded.
- Claude removed from the stack and from OMP's role mapping.
- Secrets stripped: `.env` keys, credentials, tokens, databases, sessions, logs.

## Left to the builder

- Writing the actual README and notes.
- Verifying no secret material remains.
- Pushing the branch and confirming the remote repository.
