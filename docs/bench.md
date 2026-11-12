# Benchmark local models

`inferoid bench` runs one task through Claude Code and Ollama in an isolated copy of a repository. The original checkout is not edited. The report sent to Forge contains metrics only: it does not contain the brief, source, file contents, model output, paths or usernames.

## Run the same task on several models

Put the task instructions in a brief file and choose a check command that returns zero only when the task is complete:

```sh
inferoid bench run --repo . --brief tasks/fix.md --check "uv run --with pytest pytest -q" \
  --pack my-pack --task fix-v1 --model nemotron-3.5-bolt:e4b
inferoid bench run --repo . --brief tasks/fix.md --check "uv run --with pytest pytest -q" \
  --pack my-pack --task fix-v1 --model gpt-oss:20b
```

Run them one at a time. The runner loads one model, samples Ollama memory and swap use, runs the check in the isolated copy, and removes that copy when finished. Add `--keep` when you need to inspect the worktree. Use `--no-submit` to keep a result local.

To add a task, add a brief file and a shell check command. Use a stable pack name and task id so Forge can compare runs of the same task. Keep private details out of pack and task names because they are included in the metrics report.

## Read results

The terminal summary and the JSON file under `~/.inferoid/bench/reports/` show pass or fail, wall time, first reply time, turns, tool calls, tokens, peak model memory, swap and memory pressure. Failure kinds distinguish a failed check, context exhaustion, memory failure, tool format error, no output, timeout and other engine errors.

```sh
inferoid bench list
```

Forge verdicts are `good`, `chance` or `bad`. `good` means the check passed within the machine budget. `chance` identifies a context or memory limit that may be improved with a lower context or a different model build. `bad` identifies a check or tool-quality failure, or a model that is far beyond the available budget.

## Compare with a cloud reference

Create a JSON file containing only the cloud run's `model`, `status`, `wall_s` and `tool_calls`, then pass it to the local run:

```sh
inferoid bench run --repo . --brief tasks/fix.md --check "uv run --with pytest pytest -q" \
  --pack my-pack --task fix-v1 --reference cloud-result.json
```

The reference is copied into the report through the same field allowlist.

## Forge sharing and memory guards

```sh
inferoid forge login --token YOUR_TOKEN
inferoid forge status
inferoid forge logout
inferoid bench submit ~/.inferoid/bench/reports/REPORT.json
```

Sharing is opt-in. The token is stored with mode `0600`, is never printed, and submission failures leave the local report in place. A model that `memory_problem()` says cannot fit safely is refused unless `--force` is supplied. Keep one model running at a time and respect the context cap selected for the machine.

## What gets benchmarked

By default the copy is your current working tree: the last commit plus uncommitted changes and untracked files that git does not ignore. Use `--from-commit` to benchmark the last commit only. Ignored folders (audio, analysis caches, datasets) are not copied; link them in with `--share`:

```sh
inferoid bench run --repo ~/dev/myproject --brief path/to/brief.md \
  --check "python path/to/check.py" \
  --pack myproject --task task-id --kind plan \
  --share data --model gemma4:12b-it-qat
```

## Writing a good check

The check decides pass or fail, so make it strict: it must fail on unchanged input (a copy of the starting files) and pass only when the task's real goal is met. A check that accepts the untouched starting state measures nothing.

## Bigger context

The default context is set by your Mac's memory (32K on 24 GB). A model that is smaller than the one that sets that default can use more. `--context max` loads the model at 128K, 96K, 64K, 48K and 32K in turn, reads how much memory Ollama actually reserved, and keeps the largest that leaves 2 GB free inside macOS's default GPU budget. `--context 64k` asks for one size and refuses if it does not fit. The same option works for `claude-local`.

Expect a longer first reply with a bigger context: the model reads everything you give it at a few hundred tokens a second.

## Context and transcripts

inferoid creates an Ollama tag with your machine's context cap built in (for example `gemma4-12b-it-qat-ctx32k`), because some models, such as `hf.co/...` imports, load with a 4,096-token context otherwise. If Ollama still loads a smaller context, the run stops before starting and is reported as an engine error, not as a model failure.

Each run keeps the full Claude Code transcript on your machine (`~/.inferoid/bench/transcripts/`, readable only by you) so you or your agent can see why a model failed. Transcripts are never sent to Forge.

Run one model at a time: wait for a run to finish (or press Ctrl+C) before starting the next.
