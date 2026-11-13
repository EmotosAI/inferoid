"""`inferoid setup`: pick a local model for this machine, download it, and connect it to your coding agent."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

from . import claude, sandbox
from .night import machine_memory

MIN_CLAUDE = (2, 1, 242)  # --settings modelPicker
LAUNCHER = "claude-local"
CHOICES = [
    ("nemotron-3.5-lightning:30b-a3b-mlx", 24, "fastest in our 24 GB handover test; this MLX build reads and writes faster on Apple chips"),
    ("gpt-oss:20b", 24, "second in the same test"),
    ("nemotron-3.5-bolt:e4b", 8, "for 16 GB Macs"),
]
LAB = "https://inferoid.com/lab"
# Workloads asked at setup. Coding models are the ones measured on a real repository task on a 24 GB Apple M4
# (inferoid.com/lab/free-agents, 13 Nov 2026): (model, minimum RAM GB, measured result, context it was measured with).
WORKLOADS = {
    "coding": {
        "label": "Coding agent: edits, tests and tool calls in your repos (Claude Code, Codex)",
        "link": LAB + "/free-agents",
        "models": [
            ("hf.co/sleepyeldrazi/Ornith-1.5-REAP-20B-A3B-GGUF:UD-IQ4_NL", 24, "passed 2/2 on our repo task, about 5 min each", 65536),
            ("hf.co/crucible-labs/Qwen3.6-35B-A3B-REAP-48-v2-GGUF", 24, "passed 2/2 on our repo task, about 7 min each", 98304),
            ("hf.co/sleepyeldrazi/Nemotron-3.5-Lightning-30B-A3B-REAP-20B-LoRA-IQ4NL:IQ4_NL", 24, "passed 1/2, fastest at writing (about 32 tokens/s)", 65536),
        ],
    },
    "general": {
        "label": "General: chat, writing, quick questions",
        "link": LAB,
        "models": [(m, ram, why + " (not benchmarked for general use yet)", 0) for m, ram, why in CHOICES],
    },
}
MODEL_GB = claude.MODEL_GB

SUPERVISOR_SNIPPET = [
    "When to delegate: multi-step mechanical edits, test fixing, or data wrangling with a checkable result.",
    "How to brief: give one concrete goal, the exact files, and a shell check that proves success.",
    "After it returns: review the diff and run the check yourself before accepting the work.",
    "Never delegate secrets, credentials, private keys, or files containing them.",
    "Put these five lines in AGENTS.md or CLAUDE.md so the supervisor knows the local path.",
]


def config_path() -> Path:
    return claude.state_dir().parent / "config.json"


class Prompter:
    """input() with defaults; tests pass canned answers keyed by question id."""

    def __init__(self, answers: dict[str, str] | None = None, out: Callable[[str], None] = print):
        self.answers, self.out = answers, out

    def ask(self, key: str, question: str, default: str) -> str:
        if self.answers is not None:
            return str(self.answers.get(key, default))
        reply = input(f"{question} [{default}]: ").strip()
        return reply or default

    def yes(self, key: str, question: str, default: bool = True) -> bool:
        reply = self.ask(key, question + (" (Y/n)" if default else " (y/N)"), "y" if default else "n").lower()
        return reply.startswith("y")


def _version(text: str) -> tuple[int, ...]:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else ()


def _ollama_models() -> set[str] | None:
    try:
        return {m.get("name") for m in claude._ollama("/api/tags").get("models", [])}
    except OSError:
        return None


def _build_report() -> str:
    import contextlib
    import io

    from .parsers import format_scan, scan
    from .reporting import report

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(format_scan(scan(30)))
        report(None)
    return buf.getvalue().strip() + "\nFull report: ~/.inferoid/report.html"


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".inferoid-bak"))


def configure_claude(path: Path | None = None, out: Callable[[str], None] = print,
                     runner: Callable[..., Any] | None = None) -> bool:
    command = ["claude", "mcp", "add", "--scope", "user", "inferoid", "--", "inferoid", "mcp"]
    rendered = " ".join(command)
    if not shutil.which("claude"):
        out("Claude Code was not found. Run: " + rendered)
        return False
    try:
        result = (runner or subprocess.run)(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        out("Claude Code MCP registration failed. Run: " + rendered)
        return False
    if getattr(result, "returncode", 1) != 0:
        out("Claude Code MCP registration failed. Run: " + rendered)
        return False
    out("Claude Code now has the inferoid local_agent tool.")
    return True


def configure_codex(home: Path | None = None, out: Callable[[str], None] = print) -> bool:
    root = home or Path.home()
    path = root / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    text = path.read_text() if path.exists() else ""
    if re.search(r"(?m)^\[mcp_servers\.inferoid\]\s*$", text):
        out(f"Codex already has inferoid in {path}")
        return True
    # tool_timeout_sec: local runs take minutes. default_tools_approval_mode: `codex exec` refuses MCP calls that
    # need approval ("MCP tool call requires approval, but approval policy is never"), so local_agent is pre-approved.
    block = ('[mcp_servers.inferoid]\ncommand = "inferoid"\nargs = ["mcp"]\ntool_timeout_sec = 1000\n'
             'default_tools_approval_mode = "approve"\n')
    path.write_text(text.rstrip() + ("\n\n" if text.strip() else "") + block)
    out(f"Added inferoid to {path}")
    return True


def configure_cursor(home: Path | None = None, out: Callable[[str], None] = print) -> bool:
    root = home or Path.home()
    path = root / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as exc:
        out(f"Cursor MCP configuration is not valid JSON: {exc}")
        return False
    if not isinstance(data, dict):
        data = {}
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        out(f"Cursor MCP configuration has a non-object mcpServers in {path}")
        return False
    if "inferoid" not in servers:
        servers["inferoid"] = {"command": "inferoid", "args": ["mcp"]}
        path.write_text(json.dumps(data, indent=2) + "\n")
        out(f"Added inferoid to {path}")
    else:
        out(f"Cursor already has inferoid in {path}")
    return True


def configure_supervisors(supervisor: str, home: Path | None = None,
                          out: Callable[[str], None] = print) -> None:
    if supervisor not in {"claude", "codex", "cursor", "all"}:
        raise ValueError("supervisor must be claude, codex, cursor, or all")
    targets = ("claude", "codex", "cursor") if supervisor == "all" else (supervisor,)
    if "claude" in targets:
        configure_claude(out=out)
    if "codex" in targets:
        configure_codex(home=home, out=out)
    if "cursor" in targets:
        configure_cursor(home=home, out=out)
    out("\nSupervisor guidance snippet for AGENTS.md or CLAUDE.md:")
    for line in SUPERVISOR_SNIPPET:
        out(line)


def check_sandbox(out: Callable[[str], None] = print) -> str:
    """Run the sandbox self-test quietly; return verified / failed / unavailable."""
    from . import sandbox
    if not sandbox.available():
        return "unavailable"
    import contextlib
    import io
    import tempfile
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(buf):
        rc = sandbox.selftest(Path(tmp) / "wt")
    if rc != 0:
        out("\nSandbox self-test FAILED:\n" + buf.getvalue())
        return "failed"
    out("\nSafety check: sandbox self-test passed (" + str(buf.getvalue().count("PASS ")) + " checks)")
    return "verified"


def pick_for_workload(p: "Prompter", out: Callable[[str], None], mem: dict[str, float], fallback: tuple[str, float]) -> tuple[str, float, int]:
    """Ask what the local model is for, then offer up to three models that fit this machine, measured where we have
    results, plus a 4th option that links to the benchmark so the user can read it and come back."""
    out("\nWhat will the local model mostly do?")
    keys = list(WORKLOADS)
    for i, key in enumerate(keys, 1):
        out(f"  {i}. {WORKLOADS[key]['label']}")
    raw = p.ask("workload", "Pick a number", "1")
    work = WORKLOADS[keys[int(raw) - 1] if raw.isdigit() and 0 < int(raw) <= len(keys) else keys[0]]
    ram = mem.get("vram_gb") or mem.get("ram_gb", 0.0)
    fits = [x for x in work["models"] if ram >= x[1] and claude.fits_safely(x[0], MODEL_GB.get(x[0], 0.0), mem)][:3]
    if not fits:
        out(f"The models we measured for this need more memory than this machine has; offering the best fit instead.")
        fits = [(fallback[0], 0, "best fit for this memory", 0)]
    out("\nModels for this on your machine:")
    for i, (m, _, why, _ctx) in enumerate(fits, 1):
        out(f"  {i}. {claude.model_label(m)} (~{MODEL_GB.get(m, 0.0):.0f} GB): {why}")
    learn = len(fits) + 1
    out(f"  {learn}. Learn which one to pick: {work['link']}")
    pick = p.ask("model_choice", "Pick a number, or type any Ollama model name", "1")
    if pick == str(learn):
        out(f"\nOpen {work['link']} to compare them on the same task, then come back here.")
        pick = p.ask("model_choice_after_learn", f"Pick 1-{len(fits)}, or type any Ollama model name", "1")
    if pick.isdigit() and 0 < int(pick) <= len(fits):
        m, _, _, ctx = fits[int(pick) - 1]
        return m, MODEL_GB.get(m, 0.0), ctx
    return pick, MODEL_GB.get(pick, 0.0), 0


def run(project: Path, answers: dict[str, str] | None = None, out: Callable[[str], None] = print,
        pull: Callable[[str], int] | None = None, claude_version: Callable[[], str] | None = None,
        bin_dir: Path | None = None, gpu_raiser: Callable | None = None, reporter: Callable[[], str] | None = None,
        supervisor: str | None = None, sandbox_checker: Callable[[], str] | None = None) -> int:
    p = Prompter(answers, out)
    out("Inferoid setup: a free local model for the work your paid agent doesn't need to do.\n")

    # 1. machine and model
    mem = machine_memory()
    chip = platform.processor() or platform.machine()
    budget = claude.default_budget(mem)
    cap = claude.context_for(mem)
    model, gb = claude.pick_model(mem)
    if mem.get("vram_gb"):
        out(f"This machine: {chip}, {mem.get('ram_gb', 0):.0f} GB memory, GPU {mem.get('gpu', 'NVIDIA')} with {mem['vram_gb']:.0f} GB "
            f"({budget:.0f} GB usable for models).")
    else:
        out(f"This machine: {chip}, {mem.get('ram_gb', 0):.0f} GB memory ({budget:.0f} GB usable by the GPU by default).")
    out(f"Safe setup for it: models that fit that budget, and a {cap:,}-token context cap so the Mac never runs out of memory.")
    model, gb, ctx = pick_for_workload(p, out, mem, (model, gb))
    if ctx > cap:
        cap = ctx
        out(f"Context for this model: {cap:,} tokens, the size it was measured with on a 24 GB Mac.")

    # 2. Ollama and the download
    if not shutil.which("ollama"):
        out("\nOllama is not installed. Install it from https://ollama.com/download (or: brew install ollama), open it, then run inferoid setup again.")
        return 1
    have = _ollama_models()
    if have is None:
        hint = (" In WSL2, run Ollama inside WSL (curl -fsSL https://ollama.com/install.sh | sh), or turn on mirrored "
                "networking so the Windows Ollama app answers on localhost." if sandbox.is_wsl() else "")
        out("\nOllama is installed but not running. Open the Ollama app (or run: ollama serve), then run inferoid setup again." + hint)
        return 1
    if model in have:
        out(f"{model} is already downloaded.")
    elif p.yes("download", f"Download {model} now?"):
        rc = (pull or (lambda m: subprocess.call(["ollama", "pull", m])))(model)
        if rc != 0:
            out(f"Download failed. Try: ollama pull {model}")
            return 1
    else:
        out(f"Skipped. Download it later with: ollama pull {model}")
    # Ollama: context cap for this Mac, one model at a time, unload after 10 idle minutes.
    wanted_env = {"OLLAMA_CONTEXT_LENGTH": str(cap), "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "10m"}
    current = {k: os.environ.get(k) or claude._launchctl(k) for k in wanted_env}
    if current != wanted_env:
        if platform.system() == "Darwin":
            if p.yes("context", f"Set Ollama to a {cap:,}-token context, one model loaded at a time, and unload after 10 idle minutes "
                               "(restarts the Ollama app)?"):
                for key, value in wanted_env.items():
                    subprocess.call(["launchctl", "setenv", key, value])
                if not claude.restart_ollama():
                    out("Quit and reopen the Ollama app so it picks up these settings.")
        else:
            out("Start Ollama with " + " ".join(f"{k}={v}" for k, v in wanted_env.items()))

    problem = claude.memory_problem(model, gb, mem)
    if problem:
        out("\nMemory: " + problem)
        if gpu_raiser and claude.gpu_limit_needed(model, gb, mem):
            gpu_raiser(claude.gpu_limit_needed(model, gb, mem), lambda q: p.yes("gpu_limit", q, default=False))

    # 3. coding agent
    if supervisor and supervisor != "claude":
        out(f"\nSupervisor selected: {supervisor}. Claude Code handoff settings are not needed for this supervisor.")
        version = ""
    else:
        out("\nWhich coding agent do you use?\n  1. Claude Code\n  (Codex, Cursor, Hermes and OpenClaw: coming soon)")
        if p.ask("harness", "Pick a number", "1") != "1":
            out("Only Claude Code is supported in this version.")
            return 1
        version = (claude_version or (lambda: subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout))() if (claude_version or shutil.which("claude")) else ""
        if not version:
            out("Claude Code is not installed. Install it from https://code.claude.com, then run inferoid setup again.")
            return 1
        if _version(version) < MIN_CLAUDE:
            out(f"Claude Code {version.strip()} is too old for the local model list. Run: claude update")
            return 1

    # 4. settings
    if supervisor and supervisor != "claude":
        handoff, threshold, global_, takeover, report_now = False, 90.0, False, "1", False
    else:
        out("\nHandoff settings")
        handoff = p.yes("handoff", "Have Claude keep a HANDOFF.md plan (goal, steps, test command) so the local model can take over?")
        threshold = 90.0
        if handoff:
            raw = p.ask("threshold", "At what usage % should Claude update HANDOFF.md first? (5-hour or weekly limit)", "90")
            threshold = float(raw) if raw.replace(".", "", 1).isdigit() and 0 < float(raw) <= 100 else 90.0
        scope = p.ask("scope", "Apply to all projects or only this folder? (all/here)", "all").lower()
        global_ = not scope.startswith("h")
        out("When Claude reaches its limit:\n  1. Tell me, and I switch with one command (claude-local)\n"
            "  2. Switch automatically (needs a local relay that carries your Claude login; off by default, coming soon)")
        takeover = p.ask("takeover", "Pick a number", "1")
        if takeover != "1":
            out("Automatic switching is not in this version yet. Using option 1.")

        out("\nUsage report: shows where your Claude limit goes (by task and step) and what the local model saved. "
            "It reads the Claude Code and Codex logs already on this Mac; nothing leaves it.")
        report_now = p.yes("report", "Build it now?")

    # 5. apply
    applied = []
    if handoff:
        applied += claude.install(project, global_, threshold)
    bin_dir = bin_dir or Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / LAUNCHER
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m inferoid claude local "$@"\n')
    launcher.chmod(0o755)
    harness_name = "Claude Code" if not supervisor or supervisor == "claude" else supervisor.title()
    cfg: dict[str, Any] = {"model": model, "harness": harness_name.lower().replace(" ", "-"), "handoff": handoff, "threshold": threshold,
                           "scope": "all projects" if global_ else str(project), "takeover": "manual", "launcher": str(launcher),
                           "set_up_at": time.strftime("%Y-%m-%d %H:%M")}
    config_path().write_text(json.dumps(cfg, indent=2) + "\n")

    if report_now:
        try:
            out((reporter or _build_report)())
        except Exception as exc:  # the report must never break setup
            out(f"The report could not be built ({exc}). Try later: inferoid scan && inferoid report")

    # 5b. safety: prove on THIS machine that local workers are sandboxed (kernel-enforced), before anything runs unattended
    sandbox_state = (sandbox_checker or check_sandbox)()
    cfg["sandbox"] = sandbox_state
    config_path().write_text(json.dumps(cfg, indent=2) + "\n")

    # 6. summary
    out("\nYou're all set.\n")
    out(f"  Local model     {claude.model_label(model)} ({model})")
    out(f"  Coding agent    {harness_name}")
    out(f"  Handoff plan    {'on: Claude keeps HANDOFF.md' if handoff else 'off'}")
    if handoff:
        out(f"  Reminder at     {threshold:.0f}% of the 5-hour or weekly limit")
        out(f"  Applies to      {'all projects (~/.claude)' if global_ else 'this folder (' + str(project) + ')'}")
    out("  At the limit    you switch with: claude-local")
    out("  Sandbox         " + {"verified": "verified on this machine: local workers can't write outside their task folder, "
                                             "read your keys, or reach the internet",
                             "failed": "CHECK FAILED: do not run local workers unattended (see above)",
                             "unavailable": "not available: on Linux or WSL2 install bubblewrap (sudo apt install bubblewrap) "
                                            "and run inferoid setup again"}[sandbox_state])
    out("  Usage report    inferoid scan && inferoid report")
    out("\nHow to use it:")
    out("  claude            work as usual with Opus or Sonnet" + (f"; at {threshold:.0f}% Claude updates HANDOFF.md" if handoff else ""))
    out(f"  claude-local      local model on {claude.model_label(model)}, free; /model shows it as \"(local, free)\"")
    out("  inferoid claude status   usage, reset time, HANDOFF.md state")
    out("  Delegate safely  ask your supervisor to use the local_agent tool with a goal, the files and a check;")
    out("                   the work runs sandboxed in a copy and is applied only when the check passes.")
    out("  Per-project      optional .inferoid.json: {\"model\": ..., \"link\": [data folders], \"allow_bash\": [...]}")
    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        out(f"\nAdd {bin_dir} to your PATH to use claude-local:  echo 'export PATH=\"{bin_dir}:$PATH\"' >> ~/.zshrc")
    if handoff:
        out("\nRestart Claude Code so it loads the new settings.")
    chosen_supervisor = supervisor
    if chosen_supervisor is None and answers is None:
        chosen_supervisor = p.ask("supervisor", "Connect a supervisor now? (claude/codex/cursor/all)", "claude").lower()
    if chosen_supervisor:
        configure_supervisors(chosen_supervisor, out=out)
    return 0
