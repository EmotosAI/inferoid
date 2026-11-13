"""Cloud fallback for machines too small for a local coding model (under 24 GB of model memory).

When Claude reaches its limit, `claude-next` hands the HANDOFF.md plan to the fallback chosen at setup:
- codex: GPT-6 Luna in Codex, included in ChatGPT Free and Go. Passed our repository task 2/2 in under a minute.
- openrouter: Inkling 2 Small on OpenRouter's free endpoint, driven by Claude Code. Passed 2/2 in about 2 minutes.
- local: the configured local model (`claude-local`), for users who prefer to stay offline.
Results: https://inferoid.com/lab/free-agents#cloud (13 Nov 2026).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import claude

OPENROUTER_BASE = "https://openrouter.ai/api"
OPENROUTER_MODEL = "thinkingmachines/inkling-2-small:free"
CODEX_PROMPT = ("Another assistant started work in this repo and was stopped. Read HANDOFF.md, then finish every unticked "
                "step, update HANDOFF.md as you go, and run the test command it names.")


def secrets_path() -> Path:
    return claude.state_dir().parent / "secrets.json"


def save_secret(name: str, value: str) -> None:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = value
    path.write_text(json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)


def load_secret(name: str) -> str:
    try:
        return str(json.loads(secrets_path().read_text()).get(name, ""))
    except (OSError, ValueError):
        return ""


def check_openrouter_key(key: str, timeout: float = 15) -> dict[str, Any]:
    """OpenRouter's own view of the key: {"ok": bool, "free_tier": bool|None, "error": str}."""
    req = urllib.request.Request(OPENROUTER_BASE + "/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode()).get("data", {})
        return {"ok": True, "free_tier": data.get("is_free_tier"), "error": ""}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "free_tier": None, "error": f"OpenRouter said {exc.code}"}
    except (OSError, ValueError) as exc:
        return {"ok": False, "free_tier": None, "error": str(exc)}


def codex_installed() -> bool:
    return shutil.which("codex") is not None


def install_codex() -> int:
    """npm install -g @openai/codex (the official Codex CLI package)."""
    if not shutil.which("npm"):
        return 127
    return subprocess.call(["npm", "install", "-g", "@openai/codex"])


def next_command(project: Path, fallback: dict[str, Any]) -> tuple[list[str], dict[str, str] | None, str]:
    """The command `claude-next` runs for this fallback: (args, env or None for the current env, description)."""
    kind = fallback.get("kind")
    if kind == "codex":
        args = ["codex"] + (["-m", fallback["model"]] if fallback.get("model") else [])
        if (project / "HANDOFF.md").exists():
            args.append(CODEX_PROMPT)
        return args, None, "Codex (GPT-6 Luna on ChatGPT Free or Go)"
    if kind == "openrouter":
        key = load_secret("openrouter_api_key")
        model = fallback.get("model", OPENROUTER_MODEL)
        base_env = {k: v for k, v in os.environ.items() if k not in claude.PROVIDER_ENV and not k.startswith(("AWS_", "ANTHROPIC_"))}
        env = dict(base_env, ANTHROPIC_BASE_URL=OPENROUTER_BASE, ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_API_KEY="",
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=model, ANTHROPIC_DEFAULT_SONNET_MODEL=model,
                   ANTHROPIC_DEFAULT_OPUS_MODEL=model, CLAUDE_CODE_SUBAGENT_MODEL=model,
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        args = ["claude", "--model", model]
        if (project / "HANDOFF.md").exists():
            args.append(claude.HANDOFF_PROMPT)
        return args, env, f"Claude Code on {model} (OpenRouter, free)"
    return [], None, ""


def run_next(project: Path, cfg: dict[str, Any], dry_run: bool = False) -> int:
    fallback = cfg.get("fallback") or {"kind": "local"}
    if fallback.get("kind") == "local":
        return claude.local(project, cfg.get("model"), False, dry_run)
    if fallback.get("kind") == "openrouter" and not load_secret("openrouter_api_key"):
        print("No OpenRouter key saved. Run inferoid setup again and choose OpenRouter.")
        return 1
    if fallback.get("kind") == "codex" and not codex_installed():
        print("Codex is not installed. Run: npm install -g @openai/codex, then codex login")
        return 1
    args, env, label = next_command(project, fallback)
    if not args:
        print(f"Unknown fallback {fallback!r}. Run inferoid setup again.")
        return 1
    print(f"Handing off to {label}" + (", starting from HANDOFF.md" if (project / 'HANDOFF.md').exists() else "") + ".")
    if dry_run:
        print(" ".join(a if " " not in a else repr(a) for a in args))
        return 0
    return subprocess.call(args, cwd=project, env=env)
