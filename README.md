# inferoid

Keep coding when Claude hits its limit, and hand chores to a free local model. `inferoid` puts a local model on your computer, next to Claude Code (or Codex), and makes the handover clean: Claude keeps a short `HANDOFF.md` plan, and when your 5-hour or weekly limit runs out, the local model picks up from that plan.

Everything runs on your machine. inferoid never reads, stores or forwards your Claude login, has no telemetry, and only talks to the local model on `localhost`.

It works on **macOS** (Apple silicon), **Linux** (NVIDIA GPU or CPU only) and **Windows through WSL2**. Pick your section below; the commands after install are the same everywhere.

## Install

### macOS (Apple silicon)

You need [Ollama](https://ollama.com/download) (the app), [Claude Code](https://code.claude.com) 2.1.242 or newer, and Python 3.11+ with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/EmotosAI/inferoid.git
cd inferoid
uv tool install .
inferoid setup
```

- **Memory:** a Mac shares memory between CPU and GPU, and macOS lets models use about 67% of it (75% above 36 GB). Setup only offers models that fit that share.
- **Ollama limits** are set for you through `launchctl` (setup asks first, then restarts the Ollama app).
- **Sandbox:** the macOS kernel sandbox (Seatbelt), built in.
- inferoid never raises the macOS GPU memory limit by itself; `claude-local --allow-gpu-limit` offers it only when macOS would still keep at least 4 GB.

### Linux (NVIDIA GPU or CPU only)

You need Ollama, Claude Code 2.1.242 or newer, Python 3.11+ with uv, and bubblewrap for the sandbox.

```sh
curl -fsSL https://ollama.com/install.sh | sh
sudo apt install bubblewrap          # Fedora: sudo dnf install bubblewrap
git clone https://github.com/EmotosAI/inferoid.git
cd inferoid
uv tool install .
inferoid setup
```

- **Memory:** with an NVIDIA GPU, setup reads its memory with `nvidia-smi` and offers models that fit the card (its memory minus 1 GB). Without one, models run on the CPU from system RAM: slower, but they work (see [Intel CPU results](https://inferoid.com/lab/intel-cpu)).
- **Ollama limits:** setup prints the three settings to use (`OLLAMA_CONTEXT_LENGTH`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`). With the systemd service, add them with `sudo systemctl edit ollama` and restart it.
- **Sandbox:** bubblewrap. If it is missing, setup says so and local workers run unsandboxed; install it and run `inferoid setup` again.

### Windows (WSL2)

inferoid runs inside WSL2, Microsoft's built-in Linux. Most developers who use Claude Code or Codex on Windows already have it; if not, run `wsl --install` in PowerShell and restart.

Then, inside your WSL2 terminal (Ubuntu), follow the Linux steps above:

```sh
curl -fsSL https://ollama.com/install.sh | sh     # Ollama inside WSL2 uses your NVIDIA GPU through WSL's CUDA support
sudo apt install bubblewrap
git clone https://github.com/EmotosAI/inferoid.git
cd inferoid
uv tool install .
inferoid setup
```

- **Use Claude Code inside WSL2** too, so the agent, inferoid and your repositories are on the same side.
- **Already run the Ollama app on Windows?** Either install Ollama inside WSL2 as above, or turn on mirrored networking (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`) so WSL2 reaches the Windows app on `localhost`.
- **GPU check:** `nvidia-smi` inside WSL2 should list your card. If it does, setup sizes models for it.
- **Sandbox:** bubblewrap, same as Linux.

## Under 24 GB? A cloud model takes over instead

No local model we measured codes well enough below 24 GB of model memory (for example an 8-18 GB MacBook Air). There, setup offers a cloud model that picks up Claude's `HANDOFF.md` plan when you reach your limit:

| Choice | Cost | On our repository task |
|---|---|---|
| **GPT-6 Luna in Codex** (recommended) | Included in ChatGPT Free and Go ($8/month) | 2/2, under a minute each |
| **Inkling 2 Small on OpenRouter** | Free: 50 requests a day, or 1,000 a day after a one-time $10 credit (free models don't use the credit) | 2/2, about 2 minutes each |
| A small local model | Free, offline | Weaker |

How the handoff works:

1. **At 70% of your 5-hour or weekly limit** (the default for cloud fallbacks), Claude writes or updates `HANDOFF.md` while it still has room to plan well.
2. **At the limit**, run `claude-next`: Codex (Luna) or Claude Code on Inkling picks up from the plan and runs its test command.
3. **When your limit resets**, Claude reads `HANDOFF.md` and reviews what was done.

Setup offers to install Codex (`npm install -g @openai/codex`) if it's missing; sign in once with `codex login`. An OpenRouter key is asked for without echo and saved only in `~/.inferoid/secrets.json`, readable by you. With a cloud fallback your code goes to that provider, and free endpoints may keep prompts. Results: [inferoid.com/lab/free-agents](https://inferoid.com/lab/free-agents#cloud).

## What `inferoid setup` does

1. Looks at your machine (memory, and your NVIDIA GPU if you have one), then asks what the local model is for: **coding** (edits, tests and tool calls in your repos) or **general** (chat, writing). It shows up to three models that fit, each with its measured result, plus a 4th option, **Learn which one to pick**, that links to the benchmark ([coding on a 24 GB Mac](https://inferoid.com/lab/free-agents)) so you can compare and come back. For coding with 24 GB and up: Ornith 1.5 REAP-20B-A3B and Qwen3.6 35B-A3B REAP-48 (both passed 2/2 on a real repository task) and Nemotron 3.5 Lightning REAP-20B (1/2). Each coding model gets the context it was measured with (64K or 96K).
2. Downloads it with Ollama and sets safe limits: a context cap for your memory (32K up to 24 GB, 64K up to 48 GB, 128K above, or the model's measured context), one model loaded at a time, and unloading after 10 idle minutes.
3. Asks which coding agent you use (Claude Code, Codex, Cursor) and registers the `local_agent` tool so it can hand work to the local model.
4. Asks for your handoff settings: keep a `HANDOFF.md` plan (on), the usage level at which Claude must update it first (90%), and whether this applies to all projects or one folder.
5. **Safety check:** runs eight escape attempts inside the sandbox on your machine (write outside the task folder, delete a file, read an SSH key, reach the internet, ...). Every one must be blocked, and work inside the folder and calls to the local model must still succeed.
6. Offers to build your usage report: where your Claude limit goes, and what the local model saved.
7. Installs the settings and a `claude-local` command, then prints what you chose.

To update later: `git pull && uv tool install --force --reinstall .` in the cloned folder. `claude-local` prints the installed version when it starts.

## What the sandbox blocks

Every command the local model runs (including `python -c` and shell tricks) runs inside the sandbox:

| | macOS | Linux and WSL2 |
|---|---|---|
| Engine | Seatbelt (`sandbox-exec`) | bubblewrap (`bwrap`) |
| Writes | Only the task folder, temp folders and Claude Code's own settings | Same |
| Secrets | No reads of `~/.ssh`, `~/.aws`, `~/.config`, Keychains, browser profiles, any `.env` file | Same; secret folders appear empty, `.env` files appear empty |
| Network | Only `localhost` (the local model) | No network at all; the local model's port is relayed in through a Unix socket |

## Daily use

```sh
claude                    # Opus or Sonnet as usual; at 90% usage Claude updates HANDOFF.md first
claude-local              # limit reached: Claude Code on the local model, starting from HANDOFF.md
claude-next               # limit reached, under 24 GB: the cloud fallback you chose picks up HANDOFF.md
inferoid claude status    # 5-hour and weekly usage, reset times, open steps in HANDOFF.md
claude                    # limit reset: back to Claude, which reads HANDOFF.md too
```

In a `claude-local` session, Claude Code shows "claude.ai connectors are disabled": expected, because that session does not use your claude.ai login. `/model` lists the local model as "(local, free)". It starts lean: no MCP servers, skills or hooks, and six core tools (Bash, Read, Edit, Write, Glob, Grep), which cuts what the model must read before your first message from about 43K tokens to about 5K. `claude-local --full` loads everything. `claude-local --resume` continues the last full conversation instead of starting from the plan; it works, but is several times slower on a 24 GB machine because the whole conversation has to be read again.

## How it works

- **Handoff rules** go into `CLAUDE.md` (`~/.claude/CLAUDE.md` for all projects, or the project's own). Claude writes `HANDOFF.md` before changing code: a one-line goal, numbered steps naming the file, function, expected behaviour and check, the exact test command, and the next step. It ticks steps as it goes and keeps the file under 60 lines.
- **A status line** records the 5-hour and weekly usage that Claude Code already shows to status line scripts. If you had a status line, it still runs.
- **Hooks** read that number. Past your threshold, Claude is told to update `HANDOFF.md` before anything else. When a turn ends on a rate limit, inferoid notes the time.
- **`claude-local`** starts Claude Code with Ollama as its endpoint, the local model as the only `/model` option for that session, and a 64K context size, then points it at `HANDOFF.md`.
- **`inferoid report`** adds a line with how much work the local model did and what the same tokens would have cost on the paid model you use most.

`inferoid claude uninstall` (add `--global` if you chose all projects) removes the rules, status line and hooks and restores your previous status line.

## Use your agent as the boss

Keep Claude Code, Codex, or Cursor as the planner and reviewer, and hand multi-step chores to the local model with one tool call. Inferoid runs the work in an isolated worktree, checks it, and applies it only when the check passes. See [the supervisor dispatch guide](docs/dispatch.md).

**Not in this version:** switching automatically at the limit without running a command. That needs a local relay that carries your Claude login to Anthropic, which Anthropic's terms restrict for third-party tools, so it will ship as an opt-in with a clear warning, not as a default.

## Benchmark

### Benchmark your local models

Run the same task on isolated copies of your repository, collect memory and quality metrics, and optionally share metrics-only reports with Forge. See the [benchmark guide](docs/bench.md).

On a 24 GB M4, Claude started a bug-fix task, wrote its `HANDOFF.md`, and was stopped after 4 steps. Each local model then finished the task from the plan alone:

| Model | Finished from HANDOFF.md |
|---|---|
| Nemotron 3.5 Lightning 30B | 1.4 min |
| gpt-oss 20B | 2.8 min |
| Granite 4.2 8B | 11.6 min |
| Qwen 3.8 27B | 12.5 min* |

\* Shared the GPU with another job for part of the run.

One task, one run each: a smoke test, not a quality ranking. Details and caveats in [docs/handover-benchmark.md](docs/handover-benchmark.md).

## Usage report: all commands

```sh
shadow scan [--days 30]
shadow machine
shadow classify [--local-model <model>] [--endpoint http://127.0.0.1:PORT/v1]
shadow replay --steps 20 --model <model> [--tier 16|48|128] [--endpoint http://127.0.0.1:PORT/v1]
shadow report [--tier 16|48|128]
shadow export --json
inferoid claude install|uninstall|local|status
inferoid night plan|run|status|mark|schedule
```

`scan` normalizes Claude files under `~/.claude/projects/**/*.jsonl` and, when set, `$CLAUDE_CONFIG_DIR/projects/**/*.jsonl`, plus Codex files under `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`, into `~/.inferoid/shadow.db`. Unknown and malformed records are skipped and counted. The database has one row per normalized model call and one row per user turn. Each call stores input, output, cache-read, cache-write, and total context tokens; model id; tool names and coarse tool facts; final text length; request position; and subagent status. Tool facts contain only names, edit line counts/file-key hashes, and shell command first words. The `user` field defaults to `local` and is reserved for a future team schema.

`machine` reports the local operating system, memory, chip information, memory tier, context limit, suitable task classes, Ollama models, and detectable LM Studio or MLX support. The dated model-fit and price tables are packaged data resources.

`classify` uses deterministic request-level heuristics by default and also assigns every model call a deterministic step class, including reasoning-only, intermediate text, web/search, subagent/task dispatch, and unknown-tool outputs. A local Ollama model can label requests when `--local-model` is supplied; the request is sent only to `http://127.0.0.1:11434`. Use `--endpoint http://127.0.0.1:PORT/v1` for any OpenAI-compatible local server. Non-local endpoint hosts are refused.

`replay --steps N` samples local-capable steps in round-robin class order, when their normalized transcript context is available and fits the selected tier. The replay system prompt is deliberately neutral because provider system prompts are absent from the logs. Available tool names are listed; full tool definitions and tool-result bodies are not invented when absent. Dispatches are scored objectively as same action, plausible, or different using the tool and key argument. Brief edits are checked when the normalized evidence permits it, otherwise judged. Short text and other non-dispatch outputs use a local judge and are labeled `judged, not measured`. Timeout and malformed responses are recorded as failed. Wall time includes the local replay request and, for judged classes, the judge request when the endpoint reports duration. A class-level replay claim is established only at N >= 20.

`report --tier 16|48|128` writes a self-contained `~/.inferoid/report.html`, prints an aggregate terminal summary, and writes `~/.inferoid/share-card.md`. Tier limits are 16k context for 16-32 GB, 32k for 48-64 GB, and 64k for 128 GB+; these are conservative because prefill time and KV-cache memory grow with context on Apple Silicon. The report keeps request-level numbers, but the primary share is step-level and includes calls, known-model API list-price equivalent dollars, and context buckets under 16k, 32k, 64k, and 128k+. The share card contains only aggregate percentages, estimated dollars, counts, context buckets, and machine details. `export --json` is a local diagnostic export containing normalized metrics and user-turn text; treat it as sensitive and do not post it publicly.

## Limits of the estimates

Dollar values use the dated public list-price baselines packaged with the application, with input, output, cache-read, and cache-write token fields where available. They are estimates, not invoices. Unknown models are not assigned a default price. Subscription users pay a flat plan, so dollars are API list-price equivalents. Request-level local share remains a heuristic based on classified task class and brief or moderate size. Step-level local share is a heuristic based on the deterministic step class and tier context limit; it is not a claim that a local model will produce equivalent code. Unclassified, reasoning-only, intermediate-text, web/search, subagent, unknown-tool, large, and over-limit steps do not count. Replay is evidence on a sampled subset, not a benchmark; judged outcomes are not measured equivalence. Missing, encrypted, compacted, provider-internal, and version-specific log fields may be skipped.

## Development

```sh
pytest
ruff check .
```

Tests use synthetic fixture logs created in the test suite. Real session contents must never be copied into this repository.
