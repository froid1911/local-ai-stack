# Herdr

Herdr is the terminal/agent runtime. It keeps panes alive in a background
server, restores layouts, marks each pane working / blocked / idle, and lets
agents spawn and prompt each other.

- Upstream: https://github.com/herdrdev/herdr · https://herdr.dev
- Version used here: `herdr 0.9.5`
- License: Apache-2.0

## Install

```bash
curl -fsSL https://herdr.dev/install.sh | sh
herdr
```

Or `brew install herdr` / `mise use -g herdr` / Windows installer from the
project site.

## Config

Copy `config.toml` to `~/.config/herdr/config.toml`.

This config sets:

- `catppuccin` theme, no auto light/dark switching
- sidebar width 30, agent panel scoped to all workspaces, sorted by priority
- agent labels on pane borders, prompt for new tab names off, close confirmation on
- toasts delivered inside Herdr after 1 s, sound enabled
- new panes follow the source cwd, `~/.herdr/worktrees` for worktrees
- agents resumed on session restore
- tmux-style prefix bindings plus Ctrl+Alt chords for pane/tab navigation
- three plugin actions: `herdr-bar` (Ctrl+Alt+B), `reviewr` (Ctrl+Alt+R),
  `triage` (Ctrl+Alt+G)

`plugins.json` records the installed plugin manifests and versions:
`herdr-bar` 0.14.0, `reviewr`, `triage`. The file contains absolute install
paths from the original machine; regenerate it with the plugin marketplace
rather than copying it verbatim if those paths do not exist.

## Why it matters here

OMP sessions are launched inside Herdr panes (`herdr:omp`). Herdr's agent
panel shows which local-model session is working, blocked, or done, so the
harness can be supervised without cloud telemetry.
