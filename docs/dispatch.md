# Use your agent as the boss

Inferoid lets Claude Code, Codex, or Cursor stay in charge while your Mac's local model performs a multi-step chore. The supervisor writes one request to `local_agent`, then reviews the returned diff and check result.

## Setup

Install inferoid, make sure Ollama is running with your chosen local model, then connect the supervisor you use:

```sh
inferoid setup --supervisor claude
inferoid setup --supervisor codex
inferoid setup --supervisor cursor
inferoid setup --supervisor all
```

You can also run `inferoid setup` and choose a supervisor at the end. The command writes the Codex or Cursor configuration with a backup, and asks Claude Code to register the stdio server when the `claude` command is available.

## Worked example

Opus has planned a repair and wants the local model to do the mechanical work. It calls `local_agent` with:

```json
{
  "task": "Fix the failing parser tests. Preserve the public API and add a regression test for the malformed record.",
  "files": ["src/inferoid/parsers.py", "tests/test_parsers.py"],
  "check": "uv run --with pytest pytest -q tests/test_parsers.py"
}
```

The local model works in an isolated Git worktree. Inferoid runs the check there and applies the diff to the supervisor's checkout only when the check passes. The supervisor should still inspect the diff. On a failure, timeout, or apply conflict, the worktree path is returned for inspection.

## Privacy and cost

The work and focused files stay on the Mac. The supervisor sees the brief and the summary. The local transcript stays in the local inferoid state directory. Nothing is submitted to Forge unless the command line `inferoid run` is given `--share`, which sends a metrics-only report.

The free path is any free supervisor plan plus the local model already running on the Mac. The supervisor plans and reviews; the local model handles the multi-step work.

## Good delegation briefs

Give one concrete goal, name the exact files, and provide a shell command that proves success. Delegate mechanical edits, test fixing, or data wrangling with a checkable result. Never delegate credentials, private keys, or files containing secrets.
