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
from typing import Any, Callable

from . import claude
from .night import machine_memory

MIN_CLAUDE = (2, 1, 242)  # --settings modelPicker
LAUNCHER = "claude-local"
CHOICES = [
    ("nemotron-3.5-lightning:30b-a3b", 24, "fastest in our 24 GB handover test (docs/handover-benchmark.md)"),
    ("gpt-oss:20b", 24, "second in the same test"),
    ("nemotron-3.5-bolt:e4b", 8, "for 16 GB Macs"),
]
MODEL_GB = {"nemotron-3.5-lightning:30b-a3b": 19.0, "gpt-oss:20b": 14.0, "nemotron-3.5-bolt:e4b": 7.5}


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


def run(project: Path, answers: dict[str, str] | None = None, out: Callable[[str], None] = print,
        pull: Callable[[str], int] | None = None, claude_version: Callable[[], str] | None = None,
        bin_dir: Path | None = None) -> int:
    p = Prompter(answers, out)
    out("Inferoid setup: a free local model for the work your paid agent doesn't need to do.\n")

    # 1. machine and model
    mem = machine_memory()
    chip = platform.processor() or platform.machine()
    budget = claude.gpu_budget(mem)
    model, gb = claude.pick_model(mem)
    out(f"This machine: {chip}, {mem.get('ram_gb', 0):.0f} GB memory ({budget:.0f} GB usable by the GPU).")
    out(f"Recommended model: {claude.model_label(model)} ({model}, ~{gb:.0f} GB): "
        + next((why for m, _, why in CHOICES if m == model), "best fit for this memory"))
    if not p.yes("use_recommended", "Use it?"):
        fits = [c for c in CHOICES if mem.get("ram_gb", 0) >= c[1]] or CHOICES[-1:]
        for i, (m, _, why) in enumerate(fits, 1):
            out(f"  {i}. {claude.model_label(m)} ({m}): {why}")
        pick = p.ask("model_choice", "Pick a number, or type any Ollama model name", "1")
        model = fits[int(pick) - 1][0] if pick.isdigit() and 0 < int(pick) <= len(fits) else pick
        gb = MODEL_GB.get(model, 0.0)

    # 2. Ollama and the download
    if not shutil.which("ollama"):
        out("\nOllama is not installed. Install it from https://ollama.com/download (or: brew install ollama), open it, then run inferoid setup again.")
        return 1
    have = _ollama_models()
    if have is None:
        out("\nOllama is installed but not running. Open the Ollama app (or run: ollama serve), then run inferoid setup again.")
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
    ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH") or claude._launchctl("OLLAMA_CONTEXT_LENGTH")
    if (not ctx or int(ctx) < claude.LOCAL_CONTEXT) and platform.system() == "Darwin":
        if p.yes("context", "Claude Code needs a 64K context in Ollama. Set it now and restart the Ollama app?"):
            subprocess.call(["launchctl", "setenv", "OLLAMA_CONTEXT_LENGTH", str(claude.LOCAL_CONTEXT)])
            subprocess.call(["osascript", "-e", 'quit app "Ollama"'], stderr=subprocess.DEVNULL)
            time.sleep(3)
            subprocess.call(["open", "-a", "Ollama"], stderr=subprocess.DEVNULL)
    elif not ctx or int(ctx) < claude.LOCAL_CONTEXT:
        out(f"Start Ollama with OLLAMA_CONTEXT_LENGTH={claude.LOCAL_CONTEXT} (Claude Code's own instructions are ~20K tokens).")

    # 3. coding agent
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

    # 5. apply
    applied = []
    if handoff:
        applied += claude.install(project, global_, threshold)
    bin_dir = bin_dir or Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / LAUNCHER
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m inferoid claude local "$@"\n')
    launcher.chmod(0o755)
    cfg: dict[str, Any] = {"model": model, "harness": "claude-code", "handoff": handoff, "threshold": threshold,
                           "scope": "all projects" if global_ else str(project), "takeover": "manual", "launcher": str(launcher),
                           "set_up_at": time.strftime("%Y-%m-%d %H:%M")}
    config_path().write_text(json.dumps(cfg, indent=2) + "\n")

    # 6. summary
    out("\nYou're all set.\n")
    out(f"  Local model     {claude.model_label(model)} ({model})")
    out("  Coding agent    Claude Code")
    out(f"  Handoff plan    {'on: Claude keeps HANDOFF.md' if handoff else 'off'}")
    if handoff:
        out(f"  Reminder at     {threshold:.0f}% of the 5-hour or weekly limit")
        out(f"  Applies to      {'all projects (~/.claude)' if global_ else 'this folder (' + str(project) + ')'}")
    out("  At the limit    you switch with: claude-local")
    out("\nHow to use it:")
    out("  claude            work as usual with Opus or Sonnet" + (f"; at {threshold:.0f}% Claude updates HANDOFF.md" if handoff else ""))
    out(f"  claude-local      Claude Code on {claude.model_label(model)}, free; /model shows it as \"(local, free)\"")
    out("  inferoid claude status   usage, reset time, HANDOFF.md state")
    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        out(f"\nAdd {bin_dir} to your PATH to use claude-local:  echo 'export PATH=\"{bin_dir}:$PATH\"' >> ~/.zshrc")
    hint = [n for n in claude.preflight(model, gb, mem) if n.startswith("hint:") and "iogpu" in n]
    if hint:
        out("\n" + hint[0].replace("hint: ", "Speed tip: "))
    if handoff:
        out("\nRestart Claude Code so it loads the new settings.")
    return 0
