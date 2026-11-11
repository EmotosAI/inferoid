# inferoid

Keep coding when Claude hits its limit. `inferoid` puts a free local model on your Mac, next to Claude Code, and makes the handover clean: Claude keeps a short `HANDOFF.md` plan, and when your 5-hour or weekly limit runs out, the local model picks up from that plan.

Everything runs on your machine. inferoid never reads, stores or forwards your Claude login, has no telemetry, and only talks to Ollama on `localhost`.

## Quick start

You need [Ollama](https://ollama.com/download), [Claude Code](https://code.claude.com) 2.1.242 or newer, and Python 3.11+ with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/EmotosAI/inferoid.git
cd inferoid
uv tool install .
inferoid setup
```

`inferoid setup` walks you through it:

1. Looks at your Mac and recommends a model. From 24 GB it is Nemotron 3.5 Lightning 30B (19 GB), the fastest in our [handover benchmark](docs/handover-benchmark.md). On 16 GB it is Nemotron 3.5 Bolt.
2. Downloads it with Ollama and sets Ollama's context to 64K, which Claude Code needs.
3. Asks which coding agent you use. Claude Code for now; Codex, Cursor, Hermes and OpenClaw are next.
4. Asks for your handoff settings: keep a `HANDOFF.md` plan (on), the usage level at which Claude must update it first (90%), and whether this applies to all projects or one folder.
5. Installs the settings and a `claude-local` command, then prints what you chose.

## Daily use

```sh
claude                    # Opus or Sonnet as usual; at 90% usage Claude updates HANDOFF.md first
claude-local              # limit reached: Claude Code on the local model, starting from HANDOFF.md
inferoid claude status    # 5-hour and weekly usage, reset times, open steps in HANDOFF.md
claude                    # limit reset: back to Claude, which reads HANDOFF.md too
```

In a `claude-local` session, `/model` lists the local model as "(local, free)". `claude-local --resume` continues the last full conversation instead of starting from the plan; it works, but is several times slower on a 24 GB Mac because the whole conversation has to be read again.

## How it works

- **Handoff rules** go into `CLAUDE.md` (`~/.claude/CLAUDE.md` for all projects, or the project's own). Claude writes `HANDOFF.md` before changing code: a one-line goal, numbered steps naming the file, function, expected behaviour and check, the exact test command, and the next step. It ticks steps as it goes and keeps the file under 60 lines.
- **A status line** records the 5-hour and weekly usage that Claude Code already shows to status line scripts. If you had a status line, it still runs.
- **Hooks** read that number. Past your threshold, Claude is told to update `HANDOFF.md` before anything else. When a turn ends on a rate limit, inferoid notes the time.
- **`claude-local`** starts Claude Code with Ollama as its endpoint, the local model as the only `/model` option for that session, and a 64K context size, then points it at `HANDOFF.md`.
- **`inferoid report`** adds a line with how much work the local model did and what the same tokens would have cost on the paid model you use most.

`inferoid claude uninstall` (add `--global` if you chose all projects) removes the rules, status line and hooks and restores your previous status line.

**Not in this version:** switching automatically at the limit without running a command. That needs a local relay that carries your Claude login to Anthropic, which Anthropic's terms restrict for third-party tools, so it will ship as an opt-in with a clear warning, not as a default.

## Benchmark

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

`replay --steps N` samples local-capable steps in round-robin class order, when their normalized transcript context is available and fits the selected tier. The replay system prompt is deliberately neutral because provider system prompts are absent from the logs. Available tool names are listed; full tool definitions and tool-result bodies are not invented when absent. Dispatches are scored objectively as same action, plausible, or different using the tool and key argument. Small edits are checked when the normalized evidence permits it, otherwise judged. Short text and other non-dispatch outputs use a local judge and are labeled `judged, not measured`. Timeout and malformed responses are recorded as failed. Wall time includes the local replay request and, for judged classes, the judge request when the endpoint reports duration. A class-level replay claim is established only at N >= 20.

`report --tier 16|48|128` writes a self-contained `~/.inferoid/report.html`, prints an aggregate terminal summary, and writes `~/.inferoid/share-card.md`. Tier limits are 16k context for 16-32 GB, 32k for 48-64 GB, and 64k for 128 GB+; these are conservative because prefill time and KV-cache memory grow with context on Apple Silicon. The report keeps request-level numbers, but the primary share is step-level and includes calls, known-model API list-price equivalent dollars, and context buckets under 16k, 32k, 64k, and 128k+. The share card contains only aggregate percentages, estimated dollars, counts, context buckets, and machine details. `export --json` is a local diagnostic export containing normalized metrics and user-turn text; treat it as sensitive and do not post it publicly.

## Limits of the estimates

Dollar values use the dated public list-price baselines packaged with the application, with input, output, cache-read, and cache-write token fields where available. They are estimates, not invoices. Unknown models are not assigned a default price. Subscription users pay a flat plan, so dollars are API list-price equivalents. Request-level local share remains a heuristic based on classified task class and small or medium size. Step-level local share is a heuristic based on the deterministic step class and tier context limit; it is not a claim that a local model will produce equivalent code. Unclassified, reasoning-only, intermediate-text, web/search, subagent, unknown-tool, large, and over-limit steps do not count. Replay is evidence on a sampled subset, not a benchmark; judged outcomes are not measured equivalence. Missing, encrypted, compacted, provider-internal, and version-specific log fields may be skipped.

## Development

```sh
pytest
ruff check .
```

Tests use synthetic fixture logs created in the test suite. Real session contents must never be copied into this repository.
