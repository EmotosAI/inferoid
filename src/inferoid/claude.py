"""Claude Code handover: keep HANDOFF.md current, warn Claude near the usage limit, continue on a free local model.

Nothing here touches the Claude login. Claude Code itself gives the status line the usage numbers
(rate_limits.five_hour / seven_day), hooks add a reminder for Claude to read, and `inferoid claude local`
starts a separate Claude Code session pointed at Ollama.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

RULES_BEGIN = "<!-- inferoid:handoff-rules -->"
RULES_END = "<!-- /inferoid:handoff-rules -->"
RULES = """## Handoff rules (so a free local model can take over)

- Before changing code, write HANDOFF.md in the repo root:
  1. Goal: one sentence.
  2. Steps: a numbered checklist. Each step names the exact file, the function and its signature, the expected behaviour, and how to check it.
  3. Test command: the exact command that proves the task is done.
- After each step, update HANDOFF.md: tick the step `[x]`, add one line on what changed, and name the next step.
- Keep HANDOFF.md under 60 lines. Write it for a local model that has not seen this conversation and has a short context window.
- If you are told the usage limit is close, update HANDOFF.md before doing anything else.
"""
HANDOFF_PROMPT = ("Another assistant started work in this repo and was stopped. Read HANDOFF.md, then finish every unticked "
                  "step, update HANDOFF.md as you go, and run the test command it names.")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
LOCAL_CONTEXT = 65536

# Handover model by unified memory. Picked from the 2026-11-11 handover benchmark (docs/handover-benchmark.md).
LOCAL_MODELS = [
    # (min_ram_gb, ollama model, approx GPU GB at 64K context)
    (24, "nemotron-3.5-lightning:30b-a3b", 19.0),
    (0, "nemotron-3.5-bolt:e4b", 7.5),
]
DEFAULT_THRESHOLD = 90.0
MARK = "inferoid claude"


def state_dir() -> Path:
    base = Path(os.environ.get("INFEROID_HOME", Path.home() / ".inferoid"))
    path = base / "claude"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- status line + hooks (run by Claude Code)

def statusline(stdin_text: str) -> str:
    """Save the usage numbers Claude Code hands the status line, then show the user's own status line or ours."""
    try:
        data = json.loads(stdin_text or "{}")
    except ValueError:
        data = {}
    limits = data.get("rate_limits") or {}
    usage = {"ts": time.time(), "session_id": data.get("session_id"), "cwd": data.get("cwd") or (data.get("workspace") or {}).get("current_dir")}
    for window in ("five_hour", "seven_day"):
        item = limits.get(window) or {}
        if "used_percentage" in item:
            usage[window] = {"pct": float(item["used_percentage"]), "resets_at": item.get("resets_at")}
    if limits:
        _write(state_dir() / "usage.json", usage)
    prev = _read(state_dir() / "config.json", {}).get("previous_statusline")
    if prev:
        try:
            return subprocess.run(prev, shell=True, input=stdin_text, capture_output=True, text=True, timeout=5).stdout.rstrip("\n")
        except (OSError, subprocess.SubprocessError):
            pass
    parts = [f"{label} {usage[w]['pct']:.0f}%" for w, label in (("five_hour", "5h"), ("seven_day", "7d")) if w in usage]
    model = (data.get("model") or {}).get("display_name") or ""
    return " · ".join([p for p in [model, *parts] if p])


def current_usage(now: float | None = None) -> tuple[float, str, Any]:
    """Highest live usage window: (percent, window name, resets_at). Windows past their reset time are ignored."""
    now = now or time.time()
    usage = _read(state_dir() / "usage.json", {})
    best = (0.0, "", None)
    for window in ("five_hour", "seven_day"):
        item = usage.get(window)
        if not item:
            continue
        resets = item.get("resets_at")
        if resets and float(resets) <= now:
            continue
        if item["pct"] > best[0]:
            best = (item["pct"], window, resets)
    return best


def hook(event: str, stdin_text: str) -> str:
    """UserPromptSubmit / PostToolUse: remind Claude to update HANDOFF.md once usage passes the threshold.
    StopFailure (matcher rate_limit): record that the limit was hit so `inferoid claude local` can say so."""
    try:
        data = json.loads(stdin_text or "{}")
    except ValueError:
        data = {}
    if event == "StopFailure":
        _write(state_dir() / "limit-hit.json", {"ts": time.time(), "session_id": data.get("session_id"), "cwd": data.get("cwd")})
        return ""
    threshold = float(_read(state_dir() / "config.json", {}).get("threshold", DEFAULT_THRESHOLD))
    pct, window, resets = current_usage()
    if pct < threshold:
        return ""
    level = "97" if pct >= 97 else "90"
    key = f"{data.get('session_id')}|{window}|{resets}|{level}"
    warned = _read(state_dir() / "warned.json", [])
    if key in warned:
        return ""
    _write(state_dir() / "warned.json", (warned + [key])[-200:])
    when = time.strftime("%H:%M", time.localtime(float(resets))) if resets else "later"
    name = "5-hour" if window == "five_hour" else "weekly"
    msg = (f"Usage check from inferoid: the {name} usage limit is {pct:.0f}% used (resets at {when}). Before anything else, "
           "update HANDOFF.md as the handoff rules in CLAUDE.md describe (goal, numbered steps with files, functions and checks, "
           "ticked progress, the exact test command, the next step), so a local model can continue if the limit is reached.")
    return json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": msg}})


# ---------------------------------------------------------------- install / uninstall

def _command(*args: str) -> str:
    return " ".join([f'"{sys.executable}"', "-m", "inferoid", "claude", *args])


def _is_ours(entry: dict) -> bool:
    return any(MARK in str(h.get("command", "")) for h in entry.get("hooks", []))


def upsert_rules(path: Path) -> None:
    text = path.read_text() if path.exists() else ""
    block = f"{RULES_BEGIN}\n{RULES}{RULES_END}"
    if RULES_BEGIN in text and RULES_END in text:
        start, end = text.index(RULES_BEGIN), text.index(RULES_END) + len(RULES_END)
        text = text[:start] + block + text[end:]
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def remove_rules(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    if RULES_BEGIN in text and RULES_END in text:
        start, end = text.index(RULES_BEGIN), text.index(RULES_END) + len(RULES_END)
        path.write_text((text[:start].rstrip() + "\n" + text[end:].lstrip()).strip() + "\n")


def settings_paths(project: Path, global_: bool) -> tuple[Path, Path]:
    if global_:
        home = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        return home / "settings.json", home / "CLAUDE.md"
    return project / ".claude" / "settings.local.json", project / "CLAUDE.md"


def install(project: Path, global_: bool = False, threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    settings_path, rules_path = settings_paths(project, global_)
    settings = _read(settings_path, {})
    if settings_path.exists():
        shutil.copy2(settings_path, settings_path.with_name(settings_path.name + ".inferoid-bak"))
    config = _read(state_dir() / "config.json", {})
    status = settings.get("statusLine") or {}
    if status.get("command") and MARK not in status["command"]:
        config["previous_statusline"] = status["command"]
    settings["statusLine"] = {"type": "command", "command": _command("statusline")}
    hooks = settings.setdefault("hooks", {})
    for event, matcher in (("UserPromptSubmit", None), ("PostToolUse", None), ("StopFailure", "rate_limit")):
        entries = [e for e in hooks.get(event, []) if not _is_ours(e)]
        entry: dict[str, Any] = {"hooks": [{"type": "command", "command": _command("hook", event)}]}
        if matcher:
            entry["matcher"] = matcher
        hooks[event] = entries + [entry]
    _write(settings_path, settings)
    config["threshold"] = threshold
    _write(state_dir() / "config.json", config)
    upsert_rules(rules_path)
    return [f"Handoff rules: {rules_path}", f"Status line and hooks: {settings_path}",
            f"Claude is reminded to update HANDOFF.md at {threshold:.0f}% of the 5-hour or weekly limit.",
            "Restart Claude Code in this project to load them."]


def uninstall(project: Path, global_: bool = False) -> list[str]:
    settings_path, rules_path = settings_paths(project, global_)
    settings = _read(settings_path, {})
    config = _read(state_dir() / "config.json", {})
    if MARK in str((settings.get("statusLine") or {}).get("command", "")):
        prev = config.pop("previous_statusline", None)
        if prev:
            settings["statusLine"] = {"type": "command", "command": prev}
        else:
            settings.pop("statusLine", None)
    for event in list((settings.get("hooks") or {}).keys()):
        settings["hooks"][event] = [e for e in settings["hooks"][event] if not _is_ours(e)]
        if not settings["hooks"][event]:
            del settings["hooks"][event]
    if settings.get("hooks") == {}:
        del settings["hooks"]
    _write(settings_path, settings)
    _write(state_dir() / "config.json", config)
    remove_rules(rules_path)
    return [f"Removed inferoid rules from {rules_path} and status line/hooks from {settings_path}."]


# ---------------------------------------------------------------- local session

def pick_model(memory: dict[str, float] | None = None) -> tuple[str, float]:
    from .night import machine_memory

    ram = (memory or machine_memory()).get("ram_gb", 0.0)
    for min_ram, model, gb in LOCAL_MODELS:
        if ram >= min_ram:
            return model, gb
    return LOCAL_MODELS[-1][1], LOCAL_MODELS[-1][2]


def gpu_budget(memory: dict[str, float]) -> float:
    ram, wired = memory.get("ram_gb", 0.0), memory.get("wired_limit_gb", 0.0)
    return wired if wired > 0 else ram * (0.67 if ram <= 36 else 0.75)


def _ollama(path: str) -> Any:
    with urllib.request.urlopen(OLLAMA + path, timeout=5) as resp:
        return json.loads(resp.read())


def preflight(model: str, model_gb: float, memory: dict[str, float]) -> list[str]:
    """Problems that stop the local session, plus hints that only cost speed (prefixed 'hint:')."""
    notes = []
    if not shutil.which("claude"):
        notes.append("Claude Code is not installed (https://code.claude.com).")
    try:
        names = {m.get("name") for m in _ollama("/api/tags").get("models", [])}
        if model not in names:
            notes.append(f"Model not downloaded. Run: ollama pull {model}")
    except OSError:
        notes.append("Ollama is not running. Open the Ollama app (or run: ollama serve).")
    ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH") or _launchctl("OLLAMA_CONTEXT_LENGTH")
    if not ctx or int(ctx) < LOCAL_CONTEXT:
        notes.append(f"hint: Claude Code's own instructions are ~20K tokens. Give Ollama a 64K context: "
                     f"launchctl setenv OLLAMA_CONTEXT_LENGTH {LOCAL_CONTEXT}, then quit and reopen Ollama.")
    budget = gpu_budget(memory)
    if memory.get("ram_gb", 0) and model_gb + 1.5 > budget:
        want = int(min(memory["ram_gb"] - 3, model_gb + 2) * 1024)
        notes.append(f"hint: {model} needs ~{model_gb:.0f} GB of GPU memory; macOS allows {budget:.0f} GB, so part runs on the CPU (slower). "
                     f"For full speed until the next reboot: sudo sysctl iogpu.wired_limit_mb={want}")
    return notes


def _launchctl(name: str) -> str:
    try:
        return subprocess.check_output(["launchctl", "getenv", name], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


MODEL_LABELS = {
    "nemotron-3.5-lightning:30b-a3b": "Nemotron 3.5 Lightning 30B",
    "nemotron-3.5-bolt:e4b": "Nemotron 3.5 Bolt",
    "gpt-oss:20b": "gpt-oss 20B",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def picker_settings(model: str) -> str:
    """Session-only /model list: just the local model, so no Claude model is offered against Ollama."""
    row = {"model": model, "label": f"{model_label(model)} (local, free)",
           "description": "Runs on this Mac through Ollama. Your Claude login is not used."}
    return json.dumps({"modelPicker": {"options": [row], "replaceBuiltInOptions": True}})


def local_command(project: Path, model: str, resume: bool) -> tuple[list[str], dict[str, str], str]:
    env = dict(os.environ, ANTHROPIC_BASE_URL=OLLAMA, ANTHROPIC_AUTH_TOKEN="ollama", ANTHROPIC_API_KEY="",
               CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(LOCAL_CONTEXT))
    args = ["claude", "--model", model, "--settings", picker_settings(model)]
    if resume:
        return args + ["--continue"], env, "resume"
    if (project / "HANDOFF.md").exists():
        return args + [HANDOFF_PROMPT], env, "handoff"
    return args, env, "plain"


def local(project: Path, model: str | None, resume: bool, dry_run: bool = False) -> int:
    from .night import machine_memory

    memory = machine_memory()
    picked, gb = pick_model(memory)
    model = model or _read(state_dir().parent / "config.json", {}).get("model") or picked
    gb = gb if model == picked else 0.0
    notes = preflight(model, gb, memory)
    for note in notes:
        print(note)
    if any(not n.startswith("hint:") for n in notes):
        return 1
    args, env, mode = local_command(project, model, resume)
    if mode == "plain":
        print("No HANDOFF.md here, so the local model starts without a plan. Use --resume to continue the last session instead.")
    hit = _read(state_dir() / "limit-hit.json", {})
    if hit and time.time() - hit.get("ts", 0) < 6 * 3600:
        print(f"Claude hit its usage limit at {time.strftime('%H:%M', time.localtime(hit['ts']))}.")
    with (state_dir() / "handoffs.jsonl").open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "cwd": str(project), "model": model, "mode": mode}) + "\n")
    print(f"Starting Claude Code on {model} (free, local). Your Claude login is not used. Type /exit to stop; run `claude` to go back to Claude.")
    if dry_run:
        print(" ".join(args))
        return 0
    os.chdir(project)
    os.execvpe("claude", args, env)
    return 0


def status(project: Path) -> list[str]:
    from .night import machine_memory

    lines = []
    usage = _read(state_dir() / "usage.json", {})
    for window, label in (("five_hour", "5-hour"), ("seven_day", "weekly")):
        item = usage.get(window)
        if item:
            resets = time.strftime("%a %H:%M", time.localtime(float(item["resets_at"]))) if item.get("resets_at") else "?"
            lines.append(f"{label} limit: {item['pct']:.0f}% used, resets {resets}")
    if not lines:
        lines.append("No usage numbers yet (they appear after the first Claude reply in a session with the inferoid status line).")
    handoff = project / "HANDOFF.md"
    if handoff.exists():
        age = (time.time() - handoff.stat().st_mtime) / 60
        open_steps = sum(1 for line in handoff.read_text().splitlines() if "[ ]" in line)
        lines.append(f"HANDOFF.md: updated {age:.0f} min ago, {open_steps} open steps")
    else:
        lines.append("HANDOFF.md: none in this folder")
    model, gb = pick_model(machine_memory())
    lines.append(f"Local model for this machine: {model} (~{gb:.0f} GB)")
    return lines
