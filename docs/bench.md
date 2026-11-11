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

## Data that is not in git

The benchmark runs in a copy of your last commit, so uncommitted changes and ignored files (audio, analysis caches, datasets) are not there. Commit what the task needs, and link ignored folders in with `--share`:

```sh
inferoid bench run --repo ~/dev/djmix --brief night/tasks/plan.md \
  --check "python scripts/night_checks.py plan projects/plans/latest.json" \
  --pack djmix --task transition-plan-v1 --kind plan \
  --share analysis --share pool --model gemma4:12b-it-qat
```

Run one model at a time: wait for a run to finish (or press Ctrl+C) before starting the next.
