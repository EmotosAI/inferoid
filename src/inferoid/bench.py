"""Run one isolated Claude Code task against one local Ollama model."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from . import __version__, claude, forge
from .night import gpu_budget_gb, machine_memory

SAMPLE_INTERVAL = 5.0
BASH_PATTERNS = ("python", "python3", "pytest", "ls", "cat", "head", "tail", "wc", "grep", "find", "git status", "git diff")


def home() -> Path:
    return Path(os.environ.get("INFEROID_HOME", Path.home() / ".inferoid"))


def reports_dir() -> Path:
    path = home() / "bench" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value[:100] or "task"


def report_path(model: str, task: str, when: dt.datetime | None = None) -> Path:
    when = when or dt.datetime.now(dt.UTC)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return reports_dir() / f"{stamp}-{_safe_name(model)}-{_safe_name(task)}.json"


def _ollama_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(claude.OLLAMA + path, timeout=2) as response:
        value = json.loads(response.read())
    return value if isinstance(value, dict) else {}


def _size_gb(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1024**3 if value > 1024 else float(value)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB|MiB|MB)", str(value), re.IGNORECASE)
    if not match:
        return 0.0
    number = float(match.group(1))
    return number / 1024 if match.group(2).lower().startswith("m") else number


def loaded_model_memory(model: str) -> float:
    try:
        models = _ollama_json("/api/ps").get("models", [])
        for item in models if isinstance(models, list) else []:
            if isinstance(item, dict) and item.get("name") == model:
                return _size_gb(item.get("size_vram") or item.get("size"))
    except (OSError, ValueError):
        pass
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=2, check=False)
        for line in result.stdout.splitlines():
            if model in line:
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB|MiB|MB)", line, re.IGNORECASE)
                if match:
                    return _size_gb(match.group(0))
    except (OSError, subprocess.SubprocessError):
        pass
    return 0.0


def swap_used_gb() -> float:
    if platform.system() == "Darwin":
        try:
            text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL, timeout=2)
            match = re.search(r"used\s*=\s*([0-9.]+)([MG])", text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 1024 if match.group(2).upper() == "M" else float(match.group(1))
        except (OSError, subprocess.SubprocessError):
            return 0.0
        return 0.0
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"SwapTotal", "SwapFree"}:
                values[key] = float(rest.split()[0]) / 1024**2
        return max(0.0, values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0))
    except (OSError, ValueError):
        return 0.0


def memory_pressure() -> str:
    if platform.system() == "Darwin":
        try:
            text = subprocess.check_output(["memory_pressure", "-Q"], text=True, stderr=subprocess.STDOUT, timeout=2).lower()
            if "critical" in text:
                return "critical"
            if "warn" in text or "warning" in text:
                return "warn"
            match = re.search(r"free percentage:\s*([0-9.]+)%", text)
            if match:
                value = float(match.group(1))
                return "critical" if value < 5 else "warn" if value < 15 else "normal"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return "normal"
    try:
        text = Path("/proc/pressure/memory").read_text()
        match = re.search(r"some\s+avg10=([0-9.]+)", text)
        if match:
            value = float(match.group(1))
            return "critical" if value >= 50 else "warn" if value >= 10 else "normal"
    except (OSError, ValueError):
        pass
    return "normal"


class Monitor:
    def __init__(self, model: str):
        self.model = model
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_model = 0.0
        self.peak_swap = 0.0
        self.pressure = "normal"
        self.baseline = 0.0  # system memory in use before the model loaded (set by the caller)
        self.weights = 0.0

    def sample(self) -> None:
        reported = loaded_model_memory(self.model)
        if self.baseline and self.weights and reported < self.weights * 0.9:
            # Ollama under-reports some models (gemma4); count what the system actually lost since before the load.
            reported = max(reported, claude.system_used_gb() - self.baseline)
        self.peak_model = max(self.peak_model, reported)
        self.peak_swap = max(self.peak_swap, swap_used_gb())
        levels = {"normal": 0, "warn": 1, "critical": 2}
        current = memory_pressure()
        if levels.get(current, 0) > levels.get(self.pressure, 0):
            self.pressure = current

    def _run(self) -> None:
        self.sample()
        while not self.stop.wait(SAMPLE_INTERVAL):
            self.sample()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="inferoid-bench-memory", daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)


def _allowed_tools(patterns: list[str]) -> list[str]:
    tools = list(claude.LEAN_TOOLS.split(","))
    tools.extend(f"Bash({pattern} *)" for pattern in [*BASH_PATTERNS, *patterns])
    return tools


def command_for(worktree: Path, model: str, brief: str, context: int, patterns: list[str]) -> tuple[list[str], dict[str, str]]:
    base, env, mode = claude.local_command(worktree, model, resume=False, full=False, context=context)
    # local_command supplies the exact settings and lean arguments. The benchmark replaces
    # its optional handoff/plain prompt with the task brief while keeping that launch shape.
    if mode == "handoff" and base and base[-1] == claude.HANDOFF_PROMPT:
        base = base[:-1]
    args = [*base, "-p", brief, "--output-format", "stream-json", "--verbose", "--max-turns", str(60)]
    args += ["--allowedTools", *_allowed_tools(patterns)]
    return args, env


def _tree_snapshot(root: Path) -> set[tuple[str, int, int]]:
    found: set[tuple[str, int, int]] = set()
    try:
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                try:
                    stat = path.stat()
                    found.add((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    pass
    except OSError:
        pass
    return found


def _git_repo(path: Path) -> bool:
    return (path / ".git").exists() or subprocess.run(["git", "-C", str(path), "rev-parse", "--git-dir"],
                                                        capture_output=True, check=False).returncode == 0


def make_isolated_copy(project: Path, run_id: str, share: list[str] | None = None,
                       from_commit: bool = False) -> tuple[Path, bool]:
    """Git repos get a detached worktree of HEAD: uncommitted changes and untracked/ignored files (data, audio,
    caches) are NOT in it. `share` lists project-relative paths to symlink in so the task can read them."""
    target = home() / "bench" / "wt" / run_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_repo(project):
        subprocess.run(["git", "-C", str(project), "worktree", "add", "--detach", str(target), "HEAD"],
                       capture_output=True, text=True, check=True)
        is_git = True
        if not from_commit:
            # Bring the working tree over: uncommitted tracked changes, then untracked files that git does not ignore.
            diff = subprocess.run(["git", "-C", str(project), "diff", "--binary", "HEAD"], capture_output=True, check=True).stdout
            if diff.strip():
                subprocess.run(["git", "-C", str(target), "apply", "--whitespace=nowarn"], input=diff,
                               capture_output=True, check=True)
            others = subprocess.run(["git", "-C", str(project), "ls-files", "--others", "--exclude-standard", "-z"],
                                    capture_output=True, check=True).stdout.split(b"\0")
            for rel in (o.decode() for o in others if o):
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / rel, dst)
    else:
        shutil.copytree(project, target)
        is_git = False
    for rel in share or []:
        src, dst = project / rel, target / rel
        if not src.exists():
            print(f"Note: --share {rel} does not exist in the project; skipped.")
            continue
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    return target, is_git


def dirty_paths(project: Path) -> list[str]:
    if not _git_repo(project):
        return []
    out = subprocess.run(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=False).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def remove_isolated_copy(project: Path, target: Path, git_worktree: bool) -> None:
    if git_worktree:
        subprocess.run(["git", "-C", str(project), "worktree", "remove", "--force", str(target)],
                       capture_output=True, check=False)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _terminate_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass


def _consume_stream(line: str, state: dict[str, Any], start: float) -> None:
    try:
        item = json.loads(line)
    except (TypeError, ValueError):
        if line.strip():
            state["errors"].append(line.strip())
        return
    if not isinstance(item, dict):
        return
    kind = item.get("type")
    if kind == "assistant":
        if state["first_reply_s"] is None:
            state["first_reply_s"] = max(0.0, time.monotonic() - start)
        state["turns"] += 1
        message = item.get("message") if isinstance(item.get("message"), dict) else item
        content = message.get("content", []) if isinstance(message, dict) else []
        if isinstance(content, list):
            state["tool_calls"] += sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
        usage = message.get("usage", {}) if isinstance(message, dict) else {}
        # Output is a sum over messages. Claude Code emits one event per content block with the same message id,
        # and Ollama often reports 0 output tokens there, so fall back to about 4 characters per token.
        reported = usage.get("output_tokens") if isinstance(usage, dict) else None
        guess = len(json.dumps(content)) // 4 if isinstance(content, list) and content else 0
        key = message.get("id") if isinstance(message, dict) and message.get("id") else f"turn-{state['turns']}"
        entry = state.setdefault("_outputs", {}).setdefault(key, {"reported": 0, "guess": 0})
        if isinstance(reported, (int, float)):
            entry["reported"] = max(entry["reported"], int(reported))
        entry["guess"] += guess
        total = sum(e["reported"] or e["guess"] for e in state["_outputs"].values())
        state["output_tokens"] = max(state["output_tokens"], total)
        usage = {k: v for k, v in usage.items() if k != "output_tokens"} if isinstance(usage, dict) else {}
    elif kind == "result":
        state["result_seen"] = True
        if state["first_reply_s"] is None:
            state["first_reply_s"] = max(0.0, time.monotonic() - start)
        if isinstance(item.get("num_turns"), int):
            state["turns"] = item["num_turns"]
        usage = item.get("usage", {})
        if item.get("is_error") or item.get("error"):
            state["errors"].append(str(item.get("error") or item.get("result") or "Claude Code error"))
    else:
        usage = item.get("usage", {})
    if isinstance(usage, dict):
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                state[key] = max(state[key], int(value))


def run_claude(args: list[str], env: dict[str, str], worktree: Path, model: str, timeout: int, context: int,
               transcript: Path | None = None, baseline: float = 0.0) -> dict[str, Any]:
    state: dict[str, Any] = {"first_reply_s": None, "turns": 0, "tool_calls": 0,
                             "input_tokens": 0, "output_tokens": 0, "result_seen": False, "errors": []}
    started = time.monotonic()
    monitor = Monitor(model)
    monitor.baseline, monitor.weights = baseline, (claude.weights_gb(model) if baseline else 0.0)
    lines: queue.Queue[str | None] = queue.Queue()
    process = subprocess.Popen(args, cwd=worktree, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, start_new_session=True)

    def reader() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader_thread = threading.Thread(target=reader, name="inferoid-bench-stream", daemon=True)
    reader_thread.start()
    monitor.start()
    timed_out = False
    ended = False
    while not ended:
        remaining = max(0.0, timeout - (time.monotonic() - started))
        if remaining <= 0:
            timed_out = True
            _terminate_group(process)
            break
        try:
            line = lines.get(timeout=min(0.2, remaining))
            if line is None:
                ended = process.poll() is not None
            else:
                if transcript is not None:
                    with transcript.open("a") as fh:
                        fh.write(line if line.endswith("\n") else line + "\n")
                _consume_stream(line, state, started)
        except queue.Empty:
            if process.poll() is not None and lines.empty():
                ended = True
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _terminate_group(process)
    monitor.finish()
    state.update({"timed_out": timed_out, "returncode": process.returncode,
                  "wall_s": max(0.0, time.monotonic() - started),
                  "peak_model_mem_gb": monitor.peak_model, "peak_swap_gb": monitor.peak_swap,
                  "memory_pressure": monitor.pressure})
    return state


def classify_failure(state: dict[str, Any], check_passed: bool, changed: bool, context: int) -> str:
    if state.get("timed_out"):
        return "timeout"
    text = " ".join(str(x) for x in state.get("errors", []))
    lower = text.lower()
    if any(re.search(pattern, lower) for pattern in (r"requires\s+[0-9.]+\s*(?:gib|gb)", r"out of memory",
                                                     r"connection (?:refused|reset)", r"model .*unload")) \
            or (state.get("memory_pressure") == "critical" and state.get("returncode", 0) not in (0, None)):
        return "out_of_memory"
    if any(term in lower for term in ("context", "prompt too long", "too many tokens", "maximum context")) \
            or (not state.get("result_seen") and state.get("input_tokens", 0) >= context * 0.9):
        return "context_exhausted"
    if any(term in lower for term in ("invalid tool input", "invalid tool", "tool_use", "tool input")):
        return "tool_format"
    if not changed and not state.get("tool_calls"):
        return "no_output"  # e.g. replied a few words and stopped, without a tool call or a file change
    if not check_passed:
        return "check_failed"
    return "engine_error"


def _engine_version() -> str:
    try:
        value = _ollama_json("/api/version").get("version")
        return str(value) if value else "unknown"
    except (OSError, ValueError):
        return "unknown"


def _model_info(model: str) -> dict[str, Any]:
    family = model.split(":", 1)[0]
    quant = "unknown"
    for marker in ("q2", "q3", "q4", "q5", "q6", "q8", "fp16", "bf16", "qat"):
        if marker in model.lower():
            quant = marker
            break
    return {"name": model, "size_gb": claude.MODEL_GB.get(model, 0.0), "family": family, "quant": quant}


def _machine_info(memory: dict[str, float]) -> dict[str, Any]:
    chip = platform.processor() or platform.machine() or "unknown"
    return {"chip": chip, "ram_gb": memory.get("ram_gb", 0.0),
            "gpu_budget_gb": gpu_budget_gb(memory.get("ram_gb", 0.0), memory.get("wired_limit_gb", 0.0)),
            "os": f"{platform.system()} {platform.release()}"}


def write_report(report: dict[str, Any], path: Path | None = None) -> Path:
    path = path or report_path(str(report["model"]["name"]), str(report["task"]["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(forge.allowlisted_report(report), indent=2, ensure_ascii=False) + "\n")
    return path


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("report must be a JSON object")
    return forge.allowlisted_report(value)


def run(project: Path, brief_file: Path, check: str, pack: str, task: str, kind: str = "coding",
        model: str | None = None, timeout: int = 1800, max_turns: int = 60,
        allow_bash: list[str] | None = None, reference: Path | None = None,
        no_submit: bool = False, force: bool = False, keep: bool = False,
        share: list[str] | None = None, from_commit: bool = False,
        context: str | None = None) -> tuple[int, Path]:
    project = project.resolve()
    brief = brief_file.read_text()
    memory = machine_memory()
    saved = forge.read_config().get("model")
    picked, picked_gb = claude.pick_model(memory)
    model = model or saved or picked
    model_gb = claude.MODEL_GB.get(model, picked_gb if model == picked else 0.0)
    if not model_gb:
        model_gb = float(_model_info(model).get("size_gb") or 0.0)  # size from Ollama for models we don't list
    dirty = dirty_paths(project) if from_commit else []
    if dirty:
        print(f"Note: {len(dirty)} uncommitted change(s) are not in the benchmark copy (it uses the last commit). "
              "Commit them first, or the task runs against the committed version.")
    problem = claude.memory_problem(model, model_gb, memory)
    if problem and not force:
        print(f"Refusing benchmark: {problem} Use --force only if you accept the memory risk.")
        return 2, Path()
    cap = claude.resolve_context(model, memory, context, force) if context else claude.context_for(memory)
    if not cap:
        return 2, Path()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    worktree: Path | None = None
    is_git = False
    state: dict[str, Any] = {"first_reply_s": None, "turns": 0, "tool_calls": 0, "input_tokens": 0,
                             "output_tokens": 0, "result_seen": False, "errors": [], "timed_out": False,
                             "wall_s": 0.0, "peak_model_mem_gb": 0.0, "peak_swap_gb": 0.0, "memory_pressure": "normal"}
    check_passed = False
    changed = False
    try:
        engine_model = claude.context_model(model, cap)
        claude._generate(engine_model, None, 0)  # start unloaded so the memory baseline excludes the model
        baseline = claude.system_used_gb()
        err = claude.ensure_loaded(engine_model)
        loaded = claude.loaded_context(engine_model) if not err else 0
        if not err and loaded and loaded < cap:
            err = (f"Ollama loaded {model} with a {loaded:,}-token context instead of {cap:,}; "
                   "the result would not say anything about the model")
            state["setup_error"] = True
        if err:
            state["errors"] = [err]
            state["setup_error"] = True
            print(f"Setup failed before the task started: {err}")
        else:
            worktree, is_git = make_isolated_copy(project, run_id, share, from_commit)
            before = _tree_snapshot(worktree)
            args, env = command_for(worktree, engine_model, brief, cap, allow_bash or [])
            # max-turns is a benchmark setting, so replace the command helper's default.
            args[args.index("--max-turns") + 1] = str(max_turns)
            transcript = home() / "bench" / "transcripts" / f"{run_id}.jsonl"
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.touch(mode=0o600)
            state = run_claude(args, env, worktree, engine_model, timeout, cap, transcript)
            print(f"Transcript (kept on this machine, never sent): {transcript}")
            after_agent = _tree_snapshot(worktree)
            changed = before != after_agent
            try:
                checked = subprocess.run(["bash", "-lc", check], cwd=worktree, capture_output=True,
                                         text=True, timeout=300, check=False)
                check_passed = checked.returncode == 0
            except subprocess.TimeoutExpired:
                state["errors"].append("check timed out")
                check_passed = False
            changed = changed or before != _tree_snapshot(worktree)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        state["errors"].append(str(exc))
    finally:
        if worktree and not keep:
            remove_isolated_copy(project, worktree, is_git)

    failure = ("engine_error" if state.get("setup_error") else
               "none" if check_passed else classify_failure(state, check_passed, changed, cap))
    if state.get("timed_out"):
        status = "timeout"
    elif check_passed:
        status = "pass"
    elif failure == "engine_error" and not state.get("result_seen"):
        status = "error"
    else:
        status = "fail"
    result = {"status": status, "check_passed": check_passed, "failure_kind": failure,
              "wall_s": round(float(state.get("wall_s", 0.0)), 3),
              "first_reply_s": round(float(state["first_reply_s"] or 0.0), 3),
              "turns": int(state.get("turns", 0)), "tool_calls": int(state.get("tool_calls", 0)),
              "input_tokens": int(state.get("input_tokens", 0)), "output_tokens": int(state.get("output_tokens", 0)),
              "peak_model_mem_gb": round(float(state.get("peak_model_mem_gb", 0.0)), 3),
              "peak_swap_gb": round(float(state.get("peak_swap_gb", 0.0)), 3),
              "memory_pressure": state.get("memory_pressure", "normal")}
    report: dict[str, Any] = {"schema": 1, "inferoid_version": __version__,
                              "submitted_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
                              "machine": _machine_info(memory), "engine": {"name": "ollama", "version": _engine_version()},
                              "model": _model_info(model), "task": {"pack": pack, "id": task, "kind": kind},
                              "settings": {"context_cap": cap, "lean": True, "effort": "low", "timeout_s": timeout},
                              "result": result}
    if reference:
        ref = json.loads(reference.read_text())
        if not isinstance(ref, dict):
            raise ValueError("reference must be a JSON object")
        report["reference"] = {key: ref[key] for key in ("model", "status", "wall_s", "tool_calls") if key in ref}
    if state.get("errors"):
        # Kept in the local file for debugging; allowlisted_report() never sends it.
        report["local_only"] = {"errors": [str(e)[:300] for e in state["errors"][:5]]}
    path = write_report(report)
    print(f"{'status':<8} {'failure_kind':<18} {'wall_s':>8} {'turns':>6} {'tools':>6}")
    print(f"{status:<8} {failure:<18} {result['wall_s']:>8.1f} {result['turns']:>6} {result['tool_calls']:>6}")
    print(f"Report: {path}")
    config = forge.read_config()
    if not no_submit and config.get("share_reports") and config.get("forge_token"):
        try:
            ok, message, verdict = forge.submit(report)
            if verdict:
                report["verdict"] = verdict
                write_report(report, path)
            print(("Forge verdict: " + json.dumps(verdict, sort_keys=True)) if ok and verdict else message)
        except (OSError, TypeError, ValueError) as exc:  # submission must never invalidate a local benchmark
            print(f"Forge submission failed ({type(exc).__name__}); the local report was kept.")
    return 0, path


def submit_files(paths: list[Path]) -> int:
    code = 0
    for path in paths:
        try:
            report = load_report(path)
            ok, message, verdict = forge.submit(report)
            if ok and verdict:
                report["verdict"] = verdict
                write_report(report, path)
            print(f"{path}: {(json.dumps(verdict, sort_keys=True) if verdict else message)}")
            if not ok:
                code = 1
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"{path}: report could not be submitted ({exc})")
            code = 1
    return code


def list_reports() -> int:
    paths = sorted(reports_dir().glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:20]
    print(f"{'model':<34} {'task':<24} {'status':<8} verdict")
    for path in paths:
        try:
            data = load_report(path)
            verdict = data.get("verdict", {}).get("kind", "-") if isinstance(data.get("verdict"), dict) else "-"
            print(f"{data['model']['name']:<34} {data['task']['id']:<24} {data['result']['status']:<8} {verdict}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return 0
