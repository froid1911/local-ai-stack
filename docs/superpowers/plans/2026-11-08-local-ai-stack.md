# Local AI stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `froid1911/local-ai-stack`, a sanitized config snapshot of the Herdr + OMP + local-inference stack.

**Architecture:** Static config monorepo. Each directory owns one layer: `herdr/` the terminal runtime, `omp/` the agent harness, `local-ai/` the inference servers and launch scripts, `docs/` the architecture and model writeup.

**Tech Stack:** Markdown, TOML, YAML, JSON, POSIX shell.

**Spec:** `docs/superpowers/specs/2026-11-08-local-ai-stack-design.md`

## Global Constraints

- Public repo, MIT license.
- Production scripts only; no experimental launch variants.
- No credentials, tokens, session databases, logs, or model weights.
- Host-specific paths are documented, not hidden.
- Claude is out of the stack.

## Review Focus

- A reader copying `omp/config.yml` hits the `tiny` role's missing `local`
  provider — README states it.
- A reader assumes `vllm` and `vllm-defect` are different ports — README states
  both point at `:8003` in the snapshot.
- A reader runs `strata-iq3_s.json` on a three-card host — GPU list was updated
  to `0,1,2`.
- A reader runs the dsv41 container while it already exists — script says it
  fails loudly.
- A reader expects weights in the repo — README and model notes say they are
  not included.

---

### Task 1: Repository scaffold

**Files:**
- Create: `README.md`, `LICENSE`, `.gitignore`

- [ ] **Step 1: Create the repo and branch**

```bash
gh repo create froid1911/local-ai-stack --public
git init -b main && git checkout -b feat/local-ai-stack
git remote add origin git@github.com:froid1911/local-ai-stack.git
```

- [ ] **Step 2: Add license and ignore rules**

Write MIT license and `.gitignore` covering `.env`, databases, logs, sessions,
weights and packs.

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE .gitignore
git commit -m "docs: scaffold local AI stack repo"
```

### Task 2: Herdr config

**Files:**
- Create: `herdr/config.toml`, `herdr/plugins.json`, `herdr/README.md`

- [ ] **Step 1: Copy the live config**

`config.toml` from `~/.config/herdr/`, `plugins.json` from the same directory.

- [ ] **Step 2: Document install, keybindings, plugins**

Note plugin install paths are machine-specific and should be regenerated.

- [ ] **Step 3: Commit**

```bash
git add herdr
git commit -m "feat: add herdr config and notes"
```

### Task 3: OMP config

**Files:**
- Create: `omp/config.yml`, `omp/models.yml`, `omp/AGENTS.md`, `omp/env.example`, `omp/README.md`

- [ ] **Step 1: Copy configs and strip Claude roles**

Remove the `advisor: anthropic/...` role and `disabledProviders: anthropic`.

- [ ] **Step 2: Document providers, roles, env**

State the `tiny` missing-provider caveat and the duplicate `:8003` caveat.

- [ ] **Step 3: Commit**

```bash
git add omp
git commit -m "feat: add omp config and notes"
```

### Task 4: Local inference scripts

**Files:**
- Create: `local-ai/strata/start-strata-iq3s.sh`, `start-strata.sh`,
  `strata-udiq4xs.sh`, `start-dsv41-container.sh`, three `strata-*.json`
- Create: `local-ai/vllm/start-27b-w4a16.sh`, `local-ai/llama.cpp/start-lfm3.sh`,
  `local-ai/stop-inference.sh`, `local-ai/README.md`

- [ ] **Step 1: Copy production scripts and JSON configs**

Update GPU lists from `[0,1,3]` to `[0,1,2]`; remove Tailscale
`allowed_hosts` from `strata-iq3_s.json`.

- [ ] **Step 2: Add the container launcher**

Reproduce the `strata-dsv41` `docker run` command with host network, `$HOME`
bind mount, GPUs 0,1,2, port 18080.

- [ ] **Step 3: Commit**

```bash
git add local-ai
git commit -m "feat: add production inference scripts and configs"
```

### Task 5: Documentation

**Files:**
- Create: `README.md`, `docs/architecture.md`, `docs/models.md`

- [ ] **Step 1: Write hardware, ports, model and benchmark tables**

- [ ] **Step 2: Commit**

```bash
git add README.md docs
git commit -m "docs: describe stack and models"
```

### Task 6: Sanitize, verify, push

- [ ] **Step 1: Scan for secrets**

```bash
grep -RInE 'gho_|sk-|API_KEY=|BEGIN (RSA|OPENSSH|PRIVATE)|credentials' \
  --exclude-dir=.git .
```

Expected: only `env.example` placeholder text.

- [ ] **Step 2: Syntax-check scripts and JSON**

```bash
for f in local-ai/**/*.sh local-ai/*.sh; do sh -n "$f"; done
python3 -m json.tool local-ai/strata/strata-iq3_s.json >/dev/null
```

- [ ] **Step 3: Push and verify**

```bash
git push -u origin feat/local-ai-stack
git push origin feat/local-ai-stack:main
gh repo view froid1911/local-ai-stack
```

Expected: public repo with `main` containing the files.
