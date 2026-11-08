# Personal agent notes

## Git workflow

- Always work on a fix or feat branch (e.g. `fix/<topic>`, `feat/<topic>`), never directly on the default branch.
- In the powerhouse repo, create worktrees with `scripts/new-worktree.sh <name>` (optionally `[branch] [base]`): it creates `~/.worktrees/powerhouse/<name>` on `fix/<name>` from `main` and runs `pnpm install`, `pnpm build`, and `pnpm tsc`, so a fresh worktree can run tests and type checks immediately.
- Make small incremental commits as work progresses — one logical change per commit — rather than one big commit at the end.
