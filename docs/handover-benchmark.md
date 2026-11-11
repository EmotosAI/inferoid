# Handover benchmark (2026-11-11)

Question: when Claude Code stops partway through a task, can a free local model on a 24 GB Mac finish it?

## Setup

- Machine: AWS `mac-m4.metal` (Apple M4, 24 GB), the same spec as a base MacBook. GPU memory limit raised to 21 GB (`iogpu.wired_limit_mb=21504`). Ollama 0.40.16, 64K context, Claude Code 2.1.322.
- Task: a Python repo with a failing test. "Fix the discount bug in `shop/pricing.py` and add tests for a 0% and a 100% discount, then run the tests." A hidden check decides pass or fail.
- The repo's CLAUDE.md holds the handoff rules that `inferoid claude install` adds.
- Claude (Sonnet 5) started the task and was cut off after 4 steps. By then it had written an 18-line HANDOFF.md and done step 1 of 4.
- Each local model then continued in three ways:
  - **Handoff only**: a new Claude Code session that reads only HANDOFF.md. This is what `inferoid claude local` does.
  - **Resume**: the whole Claude conversation, continued with `claude --resume`.
  - **From scratch**: no Claude, no plan, just the task.

## Results

| Model | Handoff only | Resume full session | From scratch |
|---|---|---|---|
| **nemotron-3.5-lightning 30B-A3B** (default) | ✅ **1.4 min** | ✅ 7.0 min | ✅ 2.3 min |
| gpt-oss 20B | ✅ 2.8 min | ✅ 11.0 min | ❌ 3.0 min |
| granite4.2 8B | ✅ 11.6 min | ✅ 14.4 min | not run |
| qwen3.8 27B | ✅ 12.5 min* | not run | not run |

\* Shared the GPU with another run for its first 6 to 7 minutes, so its real time is lower.

After each "resume" run, Claude resumed the local model's session without an error, so the handover works in both directions.

## What it shows, and what it does not

- Starting from HANDOFF.md was 4 to 5 times faster than resuming the full conversation, with the same result. A long conversation has to be re-read on every step, and a 24 GB Mac reads about 300 to 400 tokens a second.
- nemotron-3.5-lightning was the fastest in every mode, so it is the default for 24 GB and up.
- This is one task, run once per model. It is a smoke test, not a quality benchmark. A local model will fail on harder work that Claude handles; the point is to keep simple, well-planned work moving when Claude is unavailable.
- The Claude half ran through Amazon Bedrock with an API key, not a Claude subscription. The handover mechanics are the same.

Raw rows: one JSON line per run, kept with the benchmark scripts.
