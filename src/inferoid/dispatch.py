"""Hand a task to the local model and bring back a checked change."""
from __future__ import annotations

import contextlib
import json
import datetime as dt
import io
import platform
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import bench, claude, forge


def _initial_state() -> dict[str, Any]:
    return {"first_reply_s": None, "turns": 0, "tool_calls": 0, "input_tokens": 0,
            "output_tokens": 0, "result_seen": False, "errors": [], "timed_out": False,
            "wall_s": 0.0, "peak_model_mem_gb": 0.0, "peak_swap_gb": 0.0,
            "memory_pressure": "normal"}


def brief_for(task: str, files: list[str] | None = None, check: str | None = None) -> str:
    """Build the focused prompt used by the command line and MCP entry points."""
    parts = [task]
    if files:
        parts.append("Focus files: " + ", ".join(files))
    if check:
        parts.append(f"When done, run {check} and fix until it passes.")
    return "\n\n".join(parts)


def _run_check(check: str, worktree: Path, state: dict[str, Any]) -> tuple[bool, str]:
    try:
        result = subprocess.run(["bash", "-lc", check], cwd=worktree, capture_output=True,
                                text=True, timeout=300, check=False)
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired as exc:
        state.setdefault("errors", []).append("check timed out")
        output = ((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        return False, output + "\ncheck timed out\n"


def repo_config(project: Path) -> dict[str, Any]:
    """Per-repo defaults in .inferoid.json, so a supervisor's local_agent call needs no setup knowledge:
    {"link": ["data", ".venv"], "allow_bash": [".venv/bin/python"], "deny_tools": ["Edit"], "timeout_s": 900}.
    `link` paths (often gitignored data or virtualenvs) are symlinked into the worktree and never diffed back."""
    try:
        data = json.loads((project / ".inferoid.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _pathspec(links: list[str]) -> list[str]:
    return ["--", ".", *[f":(exclude){link.strip('/')}" for link in links]]


def _mark_untracked(worktree: Path, links: list[str] | None = None) -> None:
    """Make non-ignored untracked files visible to git diff without staging their contents."""
    raw = subprocess.run(["git", "-C", str(worktree), "ls-files", "--others", "--exclude-standard", "-z",
                          *_pathspec(links or [])], capture_output=True, check=True).stdout
    paths = [item for item in raw.split(b"\0") if item]
    if paths:
        subprocess.run(["git", "-C", str(worktree), "add", "-N", "--", *[p.decode() for p in paths]],
                       capture_output=True, check=True)


def _patch_info(worktree: Path, links: list[str] | None = None) -> tuple[str, list[str], str]:
    _mark_untracked(worktree, links)
    spec = _pathspec(links or [])
    diff = subprocess.run(["git", "-C", str(worktree), "diff", "--binary", *spec], capture_output=True,
                          text=True, check=True).stdout
    names = subprocess.run(["git", "-C", str(worktree), "diff", "--name-only", *spec], capture_output=True,
                           text=True, check=True).stdout
    stat = subprocess.run(["git", "-C", str(worktree), "diff", "--stat", *spec], capture_output=True,
                          text=True, check=True).stdout.strip()
    return diff, [line for line in names.splitlines() if line], stat


def _tail(value: str, lines: int = 30) -> str:
    return "\n".join(value.splitlines()[-lines:])


def _status(state: dict[str, Any], check_given: bool, check_passed: bool) -> str:
    if state.get("timed_out"):
        return "timeout"
    if state.get("setup_error") or state.get("interrupted") or (state.get("errors") and not state.get("result_seen")):
        return "error"
    if check_given and check_passed:
        return "done"
    if check_given and not check_passed:
        return "check_failed"
    if not state.get("result_seen") or state.get("errors"):
        return "error"
    return "done"


def run_local_task(project: Path, task: str, check: str | None = None,
                   files: list[str] | None = None, model: str | None = None,
                   engine: str = "ollama", engine_url: str = claude.OPENAI_ENGINE_URL,
                   engine_model: str | None = None, context: str | None = None,
                   timeout: int = 900, max_turns: int = 200,
                   allow_bash: list[str] | None = None, deny_tools: list[str] | None = None,
                   share: bool = False, output: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Run one task in a Git worktree. The worktree is kept until a caller applies the patch."""
    started = time.monotonic()
    project = project.resolve()
    cfg = repo_config(project)
    links = [str(x) for x in cfg.get("link", []) if isinstance(x, str)]
    allow_bash = [*(allow_bash or []), *[str(x) for x in cfg.get("allow_bash", []) if isinstance(x, str)]]
    deny_tools = [*(deny_tools or []), *[str(x) for x in cfg.get("deny_tools", []) if isinstance(x, str)]]
    if isinstance(cfg.get("timeout_s"), int) and timeout == 900:
        timeout = cfg["timeout_s"]
    result: dict[str, Any] = {"status": "error", "applied": False, "files_changed": [],
                              "diffstat": "", "check_output_tail": "", "wall_s": 0.0,
                              "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                              "model": model or "", "notes": []}
    if not bench._git_repo(project):
        result["notes"] = [f"Not a Git repository: {project}"]
        result["wall_s"] = round(time.monotonic() - started, 3)
        return result
    if engine not in {"ollama", "openai"}:
        result["notes"] = ["--engine must be ollama or openai"]
        return result
    if engine == "openai" and not engine_model:
        result["notes"] = ["--engine-model is required with --engine openai"]
        return result
    if engine == "openai" and not context:
        result["notes"] = ["--context is required with --engine openai"]
        return result

    memory = bench.machine_memory()
    if engine == "openai":
        selected_model = engine_model or ""
        model_gb = claude.MODEL_GB.get(selected_model, 0.0)
        cap = claude.explicit_context(context)
        if not cap:
            result["notes"] = ["--context must be a number (for example 65536 or 64k) with --engine openai"]
            return result
    else:
        saved = forge.read_config().get("model")
        picked, picked_gb = claude.pick_model(memory)
        selected_model = model or saved or picked
        model_gb = claude.MODEL_GB.get(selected_model, picked_gb if selected_model == picked else 0.0)
        if not model_gb:
            try:
                model_gb = float(bench._model_info(selected_model).get("size_gb") or 0.0)
            except (OSError, ValueError, TypeError):
                model_gb = 0.0
        if claude.memory_problem(selected_model, model_gb, memory):
            result["notes"] = ["Refusing to run: " + claude.memory_problem(selected_model, model_gb, memory)]
            return result
        cap = claude.resolve_context(selected_model, memory, context) if context else (
            claude.saved_context(selected_model, memory) or claude.context_for(memory))
        if not cap:
            result["notes"] = ["Could not choose a safe context"]
            return result
    result["model"] = selected_model

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    worktree: Path | None = None
    is_git = False
    engine_model = selected_model
    state = _initial_state()
    check_passed = not bool(check)
    check_output = ""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            if engine == "ollama":
                engine_model = claude.context_model(selected_model, cap)
                claude._generate(engine_model, None, 0)
            baseline = claude.system_used_gb()
            if engine == "ollama":
                err = claude.ensure_loaded(engine_model)
                loaded = claude.loaded_context(engine_model) if not err else 0
                if not err and loaded and loaded < cap:
                    err = (f"Ollama loaded {selected_model} with a {loaded:,}-token context instead of {cap:,}")
                    state["setup_error"] = True
                if err:
                    state["errors"] = [err]
                    state["setup_error"] = True
            else:
                err = ""
            if not err:
                worktree, is_git = bench.make_isolated_copy(project, run_id, share=links)
                before = bench._tree_snapshot(worktree)
                args, env = bench.command_for(worktree, engine_model, brief_for(task, files, check), cap,
                                               allow_bash or [], deny_tools, engine)
                args[args.index("--max-turns") + 1] = str(max_turns)
                transcript = bench.home() / "bench" / "transcripts" / f"{run_id}.jsonl"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                transcript.touch(mode=0o600)
                from .cleanproxy import CleanProxy

                proxy = CleanProxy(claude.OLLAMA if engine == "ollama" else engine_url,
                                   mode="openai" if engine == "openai" else "anthropic",
                                   model=engine_model).start()
                env["ANTHROPIC_BASE_URL"] = proxy.url
                try:
                    if engine == "openai":
                        state = bench.run_claude(args, env, worktree, engine_model, timeout, cap, transcript,
                                                 baseline=baseline, limit=claude.default_budget(memory),
                                                 engine=engine, model_memory=model_gb)
                    else:
                        state = bench.run_claude(args, env, worktree, engine_model, timeout, cap, transcript,
                                                 baseline=baseline, limit=claude.default_budget(memory))
                finally:
                    proxy.stop()
                if not state.get("interrupted") and check:
                    check_passed, check_output = _run_check(check, worktree, state)
                changed = before != bench._tree_snapshot(worktree)
                state["changed"] = changed
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        state.setdefault("errors", []).append(str(exc))
    finally:
        if engine_model and engine == "ollama":
            try:
                claude._generate(engine_model, None, 0)
            except (OSError, subprocess.SubprocessError):
                pass

    status = _status(state, bool(check), check_passed)
    if worktree:
        try:
            diff, paths, stat = _patch_info(worktree, links)
        except (OSError, subprocess.SubprocessError) as exc:
            diff, paths, stat = "", [], ""
            state.setdefault("errors", []).append(str(exc))
        result.update({"files_changed": paths, "diffstat": stat, "_diff": diff,
                       "_worktree": str(worktree), "_is_git": is_git})
    result.update({"status": status, "check_output_tail": _tail(check_output),
                   "wall_s": round(float(state.get("wall_s", time.monotonic() - started)), 3),
                   "tool_calls": int(state.get("tool_calls", 0)),
                   "input_tokens": int(state.get("input_tokens", 0)),
                   "output_tokens": int(state.get("output_tokens", 0))})
    notes = [str(item) for item in state.get("errors", [])[:5]]
    notes.extend(result.get("notes", []))
    result["notes"] = notes
    if captured.getvalue().strip():
        result["notes"].append(captured.getvalue().strip().splitlines()[-1])
    if share and worktree:
        result["_report"] = {"schema": 1, "inferoid_version": __import__("inferoid").__version__,
                              "submitted_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
                              "machine": {"chip": platform.processor() or platform.machine(),
                                          "ram_gb": memory.get("ram_gb", 0.0),
                                          "gpu_budget_gb": claude.default_budget(memory),
                                          "os": platform.system()},
                              "engine": {"name": "openai-compatible" if engine == "openai" else "ollama",
                                         "version": "unknown"},
                              "model": {"name": selected_model, "size_gb": model_gb,
                                        "family": "", "quant": ""},
                              "task": {"pack": "run", "id": "local-agent", "kind": "coding"},
                              "settings": {"context_cap": cap, "lean": True, "effort": "low",
                                            "timeout_s": timeout, "clean": True},
                              "result": {"status": result["status"], "check_passed": result["status"] == "done",
                                         "failure_kind": {"done": "none", "error": "engine_error"}.get(result["status"], result["status"]),
                                         "wall_s": result["wall_s"], "first_reply_s": 0,
                                         "turns": 0, "tool_calls": result["tool_calls"],
                                         "input_tokens": result["input_tokens"],
                                         "output_tokens": result["output_tokens"],
                                         "peak_model_mem_gb": 0, "peak_swap_gb": 0,
                                         "memory_pressure": "normal"}}
    result["wall_s"] = round(time.monotonic() - started, 3) if not state.get("wall_s") else result["wall_s"]
    return result


def finish_task(project: Path, result: dict[str, Any], apply: bool = True) -> dict[str, Any]:
    """Apply a successful task patch, or leave its worktree for inspection."""
    worktree_value = result.pop("_worktree", None)
    is_git = bool(result.pop("_is_git", False))
    diff = result.pop("_diff", "")
    worktree = Path(worktree_value) if worktree_value else None
    if result.get("status") == "done" and apply and worktree and diff:
        applied = subprocess.run(["git", "-C", str(project.resolve()), "apply", "--3way", "-"],
                                 input=diff, text=True, capture_output=True, check=False)
        if applied.returncode == 0:
            result["applied"] = True
            bench.remove_isolated_copy(project.resolve(), worktree, is_git)
        else:
            result.setdefault("notes", []).append("Could not apply the worktree diff: " +
                                                   (applied.stderr.strip() or "git apply failed"))
            result["worktree"] = str(worktree)
    elif result.get("status") == "done" and apply:
        result["applied"] = True
        if worktree:
            bench.remove_isolated_copy(project.resolve(), worktree, is_git)
    elif worktree:
        result["worktree"] = str(worktree)
    return result


def run_task(**kwargs: Any) -> dict[str, Any]:
    project = Path(kwargs.pop("project", Path.cwd())).resolve()
    result = run_local_task(project, **kwargs)
    return finish_task(project, result, apply=True)


def local_status(engine: str = "ollama", engine_url: str = claude.OPENAI_ENGINE_URL) -> dict[str, Any]:
    memory = bench.machine_memory()
    if engine == "openai":
        model = ""
        context = claude.explicit_context("32768")
        url = engine_url.rstrip("/") + "/v1/models"
    else:
        saved = forge.read_config().get("model")
        model = saved or claude.pick_model(memory)[0]
        context = claude.saved_context(model, memory) or claude.context_for(memory)
        url = claude.OLLAMA.rstrip("/") + "/api/tags"
    reachable = False
    try:
        with urllib.request.urlopen(url, timeout=2):
            reachable = True
    except (OSError, urllib.error.URLError):
        pass
    return {"engine": engine, "model": model, "safe_context": context, "reachable": reachable}
