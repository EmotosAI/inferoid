"""Claude Code handover: keep HANDOFF.md current, warn Claude near the usage limit, continue on a free local model.

Nothing here touches the Claude login. Claude Code itself gives the status line the usage numbers
(rate_limits.five_hour / seven_day), hooks add a reminder for Claude to read, and `inferoid claude local`
starts a separate Claude Code session pointed at Ollama.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

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
MAX_REPLY_TOKENS = 8192  # per reply; about 6 minutes at 24 tokens/s
LOCAL_CONTEXT = 32768  # floor; the real cap comes from context_for(memory)

# Memory safety (2026-11-11: a 24 GB Mac froze when Nemotron grew to ~94K tokens of context with the GPU limit
# raised to 21 GB). Models are picked to fit macOS's DEFAULT GPU budget; the limit is never raised automatically.
MACOS_RESERVE_GB = 4.0  # never leave macOS less than this when the GPU limit is raised (opt-in only)
RUNTIME_GB = 2.0  # engine, buffers and compute scratch on top of weights and KV cache

# Preference order; the first that fits safely on this Mac is the default.
LOCAL_MODELS = [
    # (ollama model, approx weight GB)
    ("nemotron-3.5-lightning:30b-a3b-mlx", 19.0),  # Ollama MLX build: ~25% faster prefill, ~33% faster decode than GGUF
    ("gpt-oss:20b", 14.0),
    ("nemotron-3.5-bolt:e4b", 7.5),
]
DEFAULT_THRESHOLD = 90.0
# Earlier defaults that a newer build replaces; saved configs move to the new one.
REPLACED_DEFAULTS = {"nemotron-3.5-lightning:30b-a3b": "nemotron-3.5-lightning:30b-a3b-mlx"}
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

def context_for(memory: dict[str, float]) -> int:
    """Hard context cap by memory tier, enforced in Ollama and in Claude Code (which compacts before it)."""
    ram = memory.get("ram_gb", 0.0)
    if ram <= 24:
        return 32768
    if ram <= 48:
        return 65536
    return 131072


def kv_gb(model: str, tokens: int) -> float:
    """KV cache estimate. Nemotron Lightning: 6 attention layers x 2 KV heads x 128 dims x 2 (K,V) x 2 bytes =
    6 KB/token (NVIDIA config.json). Unknown models: assume 24 KB/token (gpt-oss-20b class)."""
    per_token = 6144 if "nemotron-3.5-lightning" in model else 24576
    return per_token * tokens / 2**30


def default_budget(memory: dict[str, float]) -> float:
    """GPU memory macOS allows by default (ignores a raised iogpu.wired_limit_mb)."""
    ram = memory.get("ram_gb", 0.0)
    return ram * (0.67 if ram <= 36 else 0.75)


def gpu_budget(memory: dict[str, float]) -> float:
    wired = memory.get("wired_limit_gb", 0.0)
    return wired if wired > 0 else default_budget(memory)


def memory_need_gb(model: str, model_gb: float, memory: dict[str, float]) -> float:
    return model_gb + kv_gb(model, context_for(memory)) + RUNTIME_GB


def fits_safely(model: str, model_gb: float, memory: dict[str, float]) -> bool:
    return memory_need_gb(model, model_gb, memory) <= default_budget(memory)


def pick_model(memory: dict[str, float] | None = None) -> tuple[str, float]:
    from .night import machine_memory

    memory = memory or machine_memory()
    for model, gb in LOCAL_MODELS:
        if fits_safely(model, gb, memory):
            return model, gb
    return LOCAL_MODELS[-1]


def _ollama(path: str) -> Any:
    with urllib.request.urlopen(OLLAMA + path, timeout=5) as resp:
        return json.loads(resp.read())


def preflight(model: str, model_gb: float, memory: dict[str, float]) -> list[str]:
    """Problems that stop the local session, plus hints that only cost speed (prefixed 'hint:')."""
    notes = []
    if not shutil.which("claude"):
        notes.append("Claude Code is not installed (https://code.claude.com).")
    try:
        names = {m.get("name", "").removesuffix(":latest") for m in _ollama("/api/tags").get("models", [])}
        if model.removesuffix(":latest") not in names:
            notes.append(f"Model not downloaded. Run: ollama pull {model}")
    except OSError:
        notes.append("Ollama is not running. Open the Ollama app (or run: ollama serve).")
    cap = saved_context(model, memory) or context_for(memory)
    ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH") or _launchctl("OLLAMA_CONTEXT_LENGTH")
    if ctx and ctx.isdigit() and int(ctx) > cap:
        notes.append(f"memory: Ollama's context is {int(ctx):,} tokens, above the safe {cap:,} for this Mac. Run inferoid setup, or: "
                     f"launchctl setenv OLLAMA_CONTEXT_LENGTH {cap}, then quit and reopen Ollama.")
    # A smaller Ollama default is fine: the session uses a -ctx copy with the context baked in.
    problem = memory_problem(model, model_gb, memory)
    if problem:
        notes.append("memory: " + problem)
    return notes


MODEL_GB = {"nemotron-3.5-lightning:30b-a3b-mlx": 19.0, "nemotron-3.5-lightning:30b-a3b": 19.0, "gpt-oss:20b": 14.0, "nemotron-3.5-bolt:e4b": 7.5}


def gpu_limit_needed(model: str, model_gb: float, memory: dict[str, float]) -> int:
    """MB to set iogpu.wired_limit_mb to so the model fits, or 0 when it already fits or cannot fit safely
    (raising would leave macOS less than MACOS_RESERVE_GB)."""
    ram = memory.get("ram_gb", 0.0)
    need = memory_need_gb(model, model_gb, memory)
    if not ram or need <= gpu_budget(memory) or ram - need < MACOS_RESERVE_GB:
        return 0
    return int(need * 1024) + 512


def memory_problem(model: str, model_gb: float, memory: dict[str, float]) -> str:
    """Empty when the model fits the current GPU budget; otherwise what to do instead."""
    ram = memory.get("ram_gb", 0.0)
    need = memory_need_gb(model, model_gb, memory)
    if not ram or need <= gpu_budget(memory):
        return ""
    if ram - need < MACOS_RESERVE_GB:
        safe = pick_model(memory)[0]
        return (f"{model} needs about {need:.0f} GB with a {context_for(memory):,}-token context, which would leave macOS less than "
                f"{MACOS_RESERVE_GB:.0f} GB on this {ram:.0f} GB Mac and can freeze it. Use {safe} instead (inferoid setup, or --model {safe}).")
    want = gpu_limit_needed(model, model_gb, memory)
    return (f"{model} needs about {need:.0f} GB and macOS gives the GPU {gpu_budget(memory):.0f} GB by default. "
            f"To allow it anyway (leaves macOS {ram - need:.0f} GB; resets on restart): claude-local --allow-gpu-limit, "
            f"or sudo sysctl iogpu.wired_limit_mb={want}")


def raise_gpu_limit(want_mb: int, ask: Callable[[str], bool]) -> bool:
    """Offer to run the sudo sysctl that lets the GPU use more memory. True when the limit is now high enough."""
    if platform.system() != "Darwin":
        return False
    gb = want_mb / 1024
    if not ask(f"Let the GPU use {gb:.0f} GB of memory now? This runs `sudo sysctl iogpu.wired_limit_mb={want_mb}` "
               "and asks for your Mac password. It resets when you restart the Mac."):
        return False
    if subprocess.call(["sudo", "sysctl", f"iogpu.wired_limit_mb={want_mb}"]) != 0:
        return False
    restart_ollama()  # Ollama reads the GPU memory limit only when it starts
    return True


def restart_ollama(wait_s: float = 30.0) -> bool:
    """Quit and reopen the Ollama app, then wait for its API. False if Ollama is not the macOS app or does not come back."""
    if platform.system() != "Darwin" or subprocess.call(["pgrep", "-x", "Ollama"], stdout=subprocess.DEVNULL) != 0:
        return False
    print("Restarting the Ollama app so it sees the new settings...")
    # A polite AppleScript quit can be refused ("User canceled"), so stop the app and its server directly.
    subprocess.call(["killall", "Ollama", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.call(["open", "-a", "Ollama"], stderr=subprocess.DEVNULL)
    end = time.time() + wait_s
    while time.time() < end:
        try:
            _ollama("/api/version")
            return True
        except OSError:
            time.sleep(1)
    return False


def warm_up(model: str, timeout: float = 300.0, num_ctx: int | None = None) -> str:
    """Load the model. No num_ctx by default, so the context baked into the Ollama tag (context_model) applies;
    passing one here would override it. Returns Ollama's error text, or '' when it loaded."""
    payload: dict[str, Any] = {"model": model, "prompt": "", "keep_alive": "10m"}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return ""
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read()).get("error", str(exc))
        except ValueError:
            return str(exc)
    except OSError as exc:
        return str(exc)


PROBE_CONTEXTS = (131072, 98304, 65536, 49152, 32768)
PROBE_MARGIN_GB = 2.0  # GPU memory left free inside macOS's default budget after the model and its context


def _generate(model: str, num_ctx: int | None, keep_alive: str | int, timeout: float = 300.0) -> str:
    """Empty generate call to load or unload a model. Returns Ollama's error text, or ''."""
    payload: dict[str, Any] = {"model": model, "prompt": "", "keep_alive": keep_alive}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    req = urllib.request.Request(OLLAMA + "/api/generate", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return ""
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read()).get("error", str(exc))
        except ValueError:
            return str(exc)
    except OSError as exc:
        return str(exc)


def loaded_size_gb(model: str) -> float:
    """Total memory Ollama reports for a loaded model: weights plus the context it reserved at load time."""
    try:
        for m in _ollama("/api/ps").get("models", []):
            if m.get("name", "").removesuffix(":latest") == model.removesuffix(":latest"):
                return float(m.get("size") or 0) / 1e9
    except (OSError, ValueError, TypeError):
        pass
    return 0.0


def probe_context(model: str, memory: dict[str, float], candidates: tuple[int, ...] = PROBE_CONTEXTS,
                  gen: Callable[..., str] = _generate, size: Callable[[str], float] = loaded_size_gb,
                  loaded: Callable[[str], int] = lambda m: loaded_context(m),
                  log: Callable[[str], None] = print,
                  used: Callable[[], float] | None = None) -> int:
    """Largest context whose measured footprint fits macOS's default GPU budget minus PROBE_MARGIN_GB.
    Ollama reserves the whole context when it loads a model, so loading and reading /api/ps measures it."""
    limit = default_budget(memory) - PROBE_MARGIN_GB
    used = used or system_used_gb
    for n in sorted(candidates, reverse=True):
        before = used()
        err = gen(model, n, "30s")
        if err:
            log(f"  {n:>7,} tokens: did not load ({err[:80]})")
            gen(model, None, 0)
            continue
        gb, got, after = size(model), loaded(model), used()
        gen(model, None, 0)  # unload before trying the next size
        effective = got if got and got < n else n
        weights = weights_gb(model)
        layout = weights + kv_estimate_gb(model, effective) + RUNTIME_GB if weights else 0.0
        if weights and gb >= weights * 0.9 and layout > gb:
            # Ollama's own figure leaves out context checkpoints and compute buffers (Nemotron REAP: 12.2 vs ~14 GB).
            log(f"  {n:>7,} tokens: Ollama reported {gb:.1f} GB; the layout estimate {layout:.1f} GB is higher, using it")
            gb = layout
        if weights and gb < weights * 0.9:
            # Some Ollama engines report only part of the footprint (gemma4: 1.1 GB for 7.2 GB of weights), and
            # system-memory deltas swing by several GB a minute apart. Use a fixed formula instead: weights + context
            # memory from the model's metadata + engine overhead. Same answer every time; a run that still goes
            # over lowers the saved context (lower_saved_context) and the run monitor stops it first.
            gb = weights + kv_estimate_gb(model, effective) + RUNTIME_GB
            log(f"  {n:>7,} tokens: Ollama reported a size below the {weights:.1f} GB of weights; "
                f"estimated {gb:.1f} GB from the model's layout")
        if gb and gb <= limit:
            log(f"  {effective:>7,} tokens: {gb:.1f} GB, fits ({limit:.1f} GB allowed)")
            return effective
        log(f"  {n:>7,} tokens: {gb:.1f} GB, too much ({limit:.1f} GB allowed)")
    return context_for(memory)


def loaded_footprint_gb(weights: float, delta: float) -> float:
    """Model footprint from the system-memory change across a load. Engines that memory-map GGUF weights leave
    them in the file cache, which the change may not show; then count the weights plus the measured extra
    (context and compute buffers). Can double count a little, which is the safe direction."""
    delta = max(0.0, delta)
    return delta if delta >= weights * 0.9 else weights + delta


def _ollama_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(OLLAMA + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def lower_saved_context(model: str, memory: dict[str, float], used: int) -> int:
    """A run went over the memory limit at `used` tokens: save the next smaller probe size so the next run is safe."""
    smaller = [n for n in PROBE_CONTEXTS if n < used]
    new = max(smaller) if smaller else min(PROBE_CONTEXTS)
    save_context(model, memory, new)
    return new


def system_wired_gb() -> float:
    """Wired + compressed memory (GB) on macOS: GPU buffers and squeezed pages, without the file cache that holds
    memory-mapped GGUF weights. Used to spot sudden growth during a run. 0 when unknown (not macOS)."""
    if platform.system() != "Darwin":
        return 0.0
    try:
        text = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        page = int(re.search(r"page size of (\d+)", text).group(1))
        pages = 0
        for label in ("Pages wired down", "Pages occupied by compressor"):
            match = re.search(label + r":\s+(\d+)", text)
            pages += int(match.group(1)) if match else 0
        return pages * page / 1e9
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return 0.0


def system_used_gb() -> float:
    """Memory in use system-wide (GB): wired + active + compressed on macOS, MemTotal - MemAvailable on Linux.
    The change across a model load is what the model really costs, whatever Ollama reports."""
    try:
        if platform.system() == "Darwin":
            text = subprocess.check_output(["vm_stat"], text=True, timeout=5)
            page = int(re.search(r"page size of (\d+)", text).group(1))
            pages = 0
            for label in ("Pages wired down", "Pages active", "Pages occupied by compressor"):
                match = re.search(label + r":\s+(\d+)", text)
                pages += int(match.group(1)) if match else 0
            return pages * page / 1e9
        values = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                values[key] = int(rest.split()[0]) * 1024
        return (values["MemTotal"] - values["MemAvailable"]) / 1e9
    except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError):
        return 0.0


def weights_gb(model: str) -> float:
    """Weight file size from Ollama's model list (GB, 0 if unknown)."""
    try:
        base = model.removesuffix(":latest")
        for m in _ollama("/api/tags").get("models", []):
            if m.get("name", "").removesuffix(":latest") == base:
                return float(m.get("size") or 0) / 1e9
    except (OSError, ValueError, TypeError):
        pass
    return 0.0


def kv_estimate_gb(model: str, tokens: int) -> float:
    """Conservative KV cache estimate from /api/show: every layer full attention, fp16. Overestimates sliding-window
    and hybrid models, which is the safe direction. Falls back to 24 KB/token when the metadata is missing."""
    try:
        req = urllib.request.Request(OLLAMA + "/api/show", json.dumps({"model": model}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read()).get("model_info") or {}
    except (OSError, ValueError):
        info = {}

    def pick(suffix: str) -> Any:
        return next((v for k, v in info.items() if k.endswith(suffix)), None)

    layers = pick(".block_count")
    kv_heads = pick(".attention.head_count_kv")
    key_len = pick(".attention.key_length") or (
        (pick(".embedding_length") or 0) // max(1, pick(".attention.head_count") or 1) if pick(".embedding_length") else None)
    value_len = pick(".attention.value_length") or key_len
    if isinstance(kv_heads, list):
        per_layer = sum(int(h or 0) for h in kv_heads)
    elif layers and kv_heads:
        per_layer = int(layers) * int(kv_heads)
    else:
        per_layer = 0
    if per_layer and key_len:
        per_token = per_layer * (int(key_len) + int(value_len)) * 2
    else:
        per_token = 24576
    return per_token * tokens / 1e9


def _probe_file() -> Path:
    return state_dir().parent / "contexts.json"


def _probe_key(model: str, memory: dict[str, float]) -> str:
    return f"{model.removesuffix(':latest')}|{round(float(memory.get('ram_gb', 0)))}"


def saved_context(model: str, memory: dict[str, float]) -> int:
    """Context measured earlier for this model on this Mac, so near-limit measurements don't flip between runs."""
    try:
        return int(json.loads(_probe_file().read_text()).get(_probe_key(model, memory), 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def save_context(model: str, memory: dict[str, float], tokens: int) -> None:
    try:
        data = json.loads(_probe_file().read_text()) if _probe_file().exists() else {}
    except (OSError, ValueError):
        data = {}
    data[_probe_key(model, memory)] = int(tokens)
    _probe_file().write_text(json.dumps(data, indent=1))


def resolve_context(model: str, memory: dict[str, float], context: str | int | None, force: bool = False) -> int:
    """'max' -> measured largest safe context; a number -> that size if it measures as safe; None -> tier default."""
    if not context:
        return saved_context(model, memory) or context_for(memory)
    if str(context).lower() in ("max", "remeasure"):
        saved = saved_context(model, memory)
        if saved and str(context).lower() == "max":
            print(f"Using the measured context for {model}: {saved:,} tokens (--context remeasure measures again)")
            return saved
        print(f"Measuring the largest context {model} can use on this Mac...")
        got = probe_context(model, memory)
        save_context(model, memory, got)
        return got
    try:
        wanted = int(str(context).lower().removesuffix("k")) * (1024 if str(context).lower().endswith("k") else 1)
    except ValueError:
        print(f"--context must be a number (e.g. 65536 or 64k) or 'max', not {context!r}")
        return 0
    got = probe_context(model, memory, candidates=(wanted,), log=lambda s: None)
    if got == wanted or force:
        return wanted
    print(f"A {wanted:,}-token context does not fit safely for {model} on this Mac. Try --context max.")
    return 0


def context_model(model: str, cap: int, run: Callable[..., Any] = subprocess.run) -> str:
    """Ollama tag with num_ctx baked in. Some models (e.g. hf.co GGUF imports) carry their own num_ctx (often 4096)
    and Claude Code never sends one, so OLLAMA_CONTEXT_LENGTH alone is not enough. Reuses the same weight blobs.
    Copies the base model's whole Modelfile (TEMPLATE, RENDERER, PARSER, stop words): a bare FROM can lose the
    chat format, which leaks thinking markers like <channel|> and breaks tool calls. Rebuilt every time (cheap),
    so a fixed base model fixes its copies too."""
    base = re.sub(r"[^a-z0-9._-]+", "-", model.lower().removeprefix("hf.co/")).strip("-.")
    name = f"{base}-ctx{cap // 1024}k"
    try:
        names = {m.get("name", "").removesuffix(":latest") for m in _ollama("/api/tags").get("models", [])}
    except OSError:
        return model
    shown = run(["ollama", "show", model, "--modelfile"], capture_output=True, text=True)
    lines = [line for line in (getattr(shown, "stdout", "") or "").splitlines()
             if not line.startswith("FROM ") and not re.match(r"PARAMETER\s+num_ctx\b", line)]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text("\n".join([f"FROM {model}", *lines, f"PARAMETER num_ctx {cap}", ""]))
        done = run(["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True)
    if getattr(done, "returncode", 1) == 0 or name in names:
        return name
    return model


def loaded_context(model: str) -> int:
    """Context length Ollama actually loaded the model with (0 if unknown)."""
    try:
        for m in _ollama("/api/ps").get("models", []):
            if m.get("name", "").removesuffix(":latest") == model.removesuffix(":latest"):
                return int(m.get("context_length") or 0)
    except (OSError, ValueError, TypeError):
        pass
    return 0


def ensure_loaded(model: str, restart: Callable[[], bool] = restart_ollama, load: Callable[[str], str] = warm_up) -> str:
    """Load the model; on a 'requires N GiB' memory error, restart Ollama once (it may hold an old limit) and retry."""
    print(f"Loading {model_label(model)}...")
    err = load(model)
    if err and "requires" in err and "GiB" in err and restart():
        err = load(model)
    return err


def _launchctl(name: str) -> str:
    try:
        return subprocess.check_output(["launchctl", "getenv", name], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


MODEL_LABELS = {
    "nemotron-3.5-lightning:30b-a3b-mlx": "Nemotron 3.5 Lightning 30B (MLX)",
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
    # disableAllHooks: plugin hooks (e.g. a SessionStart hook that tells the model to load skills) cost the local model
    # minutes of reading on every new session; the local session needs none of them.
    return json.dumps({"modelPicker": {"options": [row], "replaceBuiltInOptions": True}, "disableAllHooks": True})


# Lean start for local sessions. Measured with Claude Code 2.1.281 (2026-11-11): the normal first request carries
# ~43K tokens (49 tool definitions incl. MCP, skills), which a 24 GB Mac reads in ~2.5 min; this set carries ~4.8K.
LEAN_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
LEAN_ARGS = ["--strict-mcp-config", "--disable-slash-commands", "--tools", LEAN_TOOLS, "--effort", "low"]


def local_command(project: Path, model: str, resume: bool, full: bool = False,
                  context: int = LOCAL_CONTEXT) -> tuple[list[str], dict[str, str], str]:
    # Claude Code compacts the conversation before it reaches this, so the local context never grows past the cap.
    env = dict(os.environ, ANTHROPIC_BASE_URL=OLLAMA, ANTHROPIC_AUTH_TOKEN="ollama", ANTHROPIC_API_KEY="",
               CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(context),
               # One reply may not run away (gemma4 thought for 18K tokens without ending its turn).
               CLAUDE_CODE_MAX_OUTPUT_TOKENS=os.environ.get("INFEROID_MAX_REPLY_TOKENS", str(MAX_REPLY_TOKENS)))
    args = ["claude", "--model", model, "--settings", picker_settings(model)] + ([] if full else LEAN_ARGS)
    if resume:
        return args + ["--continue"], env, "resume"
    if (project / "HANDOFF.md").exists():
        return args + [HANDOFF_PROMPT], env, "handoff"
    return args, env, "plain"


def local(project: Path, model: str | None, resume: bool, dry_run: bool = False, full: bool = False,
          allow_gpu_limit: bool = False, context: str | None = None, clean: bool = True) -> int:
    from .night import machine_memory

    memory = machine_memory()
    picked, gb = pick_model(memory)
    saved = _read(state_dir().parent / "config.json", {}).get("model")
    if saved in REPLACED_DEFAULTS and not model:
        print(f"Using {REPLACED_DEFAULTS[saved]}, the faster build of your saved model (run inferoid setup to change).")
        saved = REPLACED_DEFAULTS[saved]
    model = model or saved or picked
    gb = MODEL_GB.get(model, gb if model == picked else 0.0)
    notes = preflight(model, gb, memory)
    want = gpu_limit_needed(model, gb, memory)
    if want and allow_gpu_limit and not dry_run:
        def ask(question: str) -> bool:
            return input(question + " (y/N) ").strip().lower() in ("y", "yes")
        if raise_gpu_limit(want, ask):
            notes = [n for n in notes if not n.startswith("memory:")]
    for note in notes:
        print(note.replace("hint: ", "").replace("memory: ", ""))
    if any(not n.startswith("hint:") for n in notes) and not dry_run:
        return 1
    cap = saved_context(model, memory) or context_for(memory)
    if context and not dry_run:
        cap = resolve_context(model, memory, context)
        if not cap:
            return 1
    if not dry_run:
        model = context_model(model, cap)  # bake the cap into the Ollama tag the session uses
        err = ensure_loaded(model)
        if err:
            print(f"Ollama could not load {model}: {err}")
            if "GiB" in err:
                print("Fix: " + (memory_problem(model, gb, memory) or "quit and reopen the Ollama app, then run claude-local again."))
            return 1
    args, env, mode = local_command(project, model, resume, full, cap)
    if mode == "plain":
        print("No HANDOFF.md here, so the local model starts without a plan. Use --resume to continue the last session instead.")
    hit = _read(state_dir() / "limit-hit.json", {})
    if hit and time.time() - hit.get("ts", 0) < 6 * 3600:
        print(f"Claude hit its usage limit at {time.strftime('%H:%M', time.localtime(hit['ts']))}.")
    with (state_dir() / "handoffs.jsonl").open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "cwd": str(project), "model": model, "mode": mode}) + "\n")
    from importlib.metadata import PackageNotFoundError, version

    try:
        ver = version("inferoid")
    except PackageNotFoundError:
        ver = "dev"
    print(f"inferoid {ver}: starting Claude Code on {model} (free, local). Your Claude login is not used. Type /exit to stop; run `claude` to go back to Claude.")
    if dry_run:
        print(" ".join(args))
        return 0
    os.chdir(project)
    if not clean:
        os.execvpe("claude", args, env)
    from .cleanproxy import CleanProxy

    proxy = CleanProxy(OLLAMA).start()
    env["ANTHROPIC_BASE_URL"] = proxy.url
    import signal

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C belongs to Claude Code while it runs
    try:
        return subprocess.call(args, env=env)
    finally:
        signal.signal(signal.SIGINT, previous)
        proxy.stop()


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
