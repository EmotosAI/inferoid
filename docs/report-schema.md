# Inferoid -> Forge bench report, schema v1

Transport: `POST {FORGE_URL}/api/v1/reports` with `Authorization: Bearer <token>` and a JSON body.
Default FORGE_URL: `https://app.inferoid.com/forge`. Response: `201 {"id": "...", "verdict": {...}}`.
Privacy rule: metrics only. No prompts, code, file contents, model outputs, paths or usernames. Task identity is a pack name and task id chosen by the task author.

```json
{
  "schema": 1,
  "inferoid_version": "0.1.4",
  "submitted_at": "2026-11-12T03:00:00Z",
  "machine": {"chip": "Apple M4", "ram_gb": 24, "gpu_budget_gb": 16.1, "os": "macOS 26.1"},
  "engine": {"name": "ollama", "version": "0.40.16"},
  "model": {"name": "gemma4:12b-it-qat", "size_gb": 7.2, "family": "gemma4", "quant": "qat-int4"},
  "task": {"pack": "djmix", "id": "transition-plan-v1", "kind": "plan"},
  "settings": {"context_cap": 32768, "lean": true, "effort": "low", "timeout_s": 1800},
  "result": {
    "status": "pass | fail | error | timeout",
    "check_passed": true,
    "failure_kind": "none | check_failed | context_exhausted | out_of_memory | tool_format | no_output | timeout | engine_error",
    "wall_s": 312.4,
    "first_reply_s": 18.2,
    "turns": 14,
    "tool_calls": 19,
    "input_tokens": 41200,
    "output_tokens": 3900,
    "peak_model_mem_gb": 9.1,
    "peak_swap_gb": 0.0,  # swap ADDED during the run (from 0.1.16); swap left from earlier does not count
    "memory_pressure": "normal | warn | critical"
  },
  "reference": {"model": "gpt-5.6-luna", "status": "pass", "wall_s": 300, "tool_calls": 19}
}
```

`reference` is optional (a cloud run of the same task, for comparison).

## Forge verdict (returned and stored per model x task x machine tier)

- `good`: check passed and peak memory within the machine's default GPU budget, swap below 0.5 GB.
- `chance`: failed with `context_exhausted` or `out_of_memory`, or passed only with swap/critical pressure, AND the model is within about 1.6x of the budget (pruning/quantization/context work could plausibly fix it). Include `suggested_next`: e.g. "prune 25-35% experts on this pack's traces", "Q3 / imatrix", "tool-ify inputs to cut context".
- `bad`: failed with `check_failed`, `tool_format` or `no_output` while memory was fine (a quality problem compression will not fix), or exceeds 1.6x the budget.
- Machine tier = RAM bucket: 16, 24, 32-36, 48, 64-96, 128+.
