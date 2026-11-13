"""Kernel-enforced sandbox for local workers on macOS (Seatbelt, via sandbox-exec).

A supervisor reviews work after it runs; it cannot stop a command that already deleted or copied something. The local
model's Claude Code process (and every command it starts, including `python -c` and shell tricks) runs inside a
profile that:
- allows writes only in the task worktree, temp folders and Claude Code's own state,
- denies reads of secrets (SSH, cloud and GPG keys, Keychains, browser profiles, ~/.config, .env files),
- denies all network except localhost (the model talks only to Ollama or a local engine on this Mac).
Linux and other systems: not supported yet; callers print a warning and run unsandboxed only if the user allows it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SECRET_DIRS = [".ssh", ".aws", ".gnupg", ".config", ".kube", ".docker", ".azure", ".netrc", ".npmrc", ".pypirc",
               ".git-credentials", "Library/Keychains", "Library/Cookies", "Library/Mail", "Library/Messages",
               "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
               "Library/Safari", "Library/Application Support/com.apple.TCC", ".inferoid/config.json",
               ".forge-tokens", ".codex", ".openrouter-key"]


def available() -> bool:
    return platform.system() == "Darwin" and shutil.which("sandbox-exec") is not None


def _q(path: str | Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(worktree: Path, home: Path | None = None, extra_write: list[Path] | None = None) -> str:
    home = (home or Path.home()).resolve()
    writable = [Path(worktree).resolve(), Path("/private/tmp"), Path("/private/var/folders"),
                home / ".claude", home / "Library/Caches", *(extra_write or [])]
    lines = [
        "(version 1)",
        "(allow default)",
        # Network: nothing leaves the machine. Local engine and Unix sockets (name resolution, IPC) only.
        "(deny network*)",
        '(allow network* (remote ip "localhost:*"))',
        # No (local ip ...) rule: combined with the unix-socket rule it let internet traffic through (tested on macOS 15).
        "(allow network* (remote unix-socket))",
        # Writes: only the worktree, temp, Claude Code's state, and device nodes.
        "(deny file-write*)",
        "(allow file-write* " + " ".join(f"(subpath {_q(p)})" for p in writable)
        + f' (regex #"^{home}/\\.claude\\.json") (literal "/dev/null") (literal "/dev/tty") (regex #"^/dev/fd/"))',
        # Secrets: no reads, even of files the model is told about.
        "(deny file-read* " + " ".join(f"(subpath {_q(home / d)})" for d in SECRET_DIRS)
        + ' (regex #"/\\.env$") (regex #"/\\.env\\.[^/]*$"))',
    ]
    return "\n".join(lines) + "\n"


def wrap(args: list[str], worktree: Path, extra_write: list[Path] | None = None) -> list[str]:
    return ["sandbox-exec", "-p", profile(worktree, extra_write=extra_write), *args]


def selftest(workdir: Path) -> int:
    """Run escape attempts inside the profile. Every attempt must fail; work inside the folder must succeed."""
    if not available():
        print("sandbox-exec is not available on this system")
        return 2
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    home = Path.home()
    canary = home / ".ssh" / "inferoid-canary"
    canary.parent.mkdir(mode=0o700, exist_ok=True)
    canary.write_text("secret\n")
    py = sys.executable
    attempts = [
        ("write outside (touch ~/escaped)", ["touch", str(home / "inferoid-escaped")], False),
        ("write outside via python -c", [py, "-c", f"open({str(home / 'inferoid-escaped2')!r}, 'w').write('x')"], False),
        ("delete outside (rm a file in home)", ["rm", "-f", str(home / ".zshrc.inferoid-test")], None),
        ("read ~/.ssh secret", ["cat", str(canary)], False),
        ("read secret via python -c", [py, "-c", f"print(open({str(canary)!r}).read())"], False),
        ("internet (curl example.com)", ["curl", "-sS", "-m", "5", "https://example.com"], False),
        ("write inside the worktree", ["touch", str(workdir / "ok.txt")], True),
        ("local engine (curl localhost)", ["curl", "-sS", "-m", "5", "-o", "/dev/null", "http://127.0.0.1:11434/"], True),
    ]
    (home / ".zshrc.inferoid-test").write_text("x\n")
    failures = 0
    for name, cmd, should_work in attempts:
        rc = subprocess.run(wrap(cmd, workdir), capture_output=True, text=True, timeout=30).returncode
        if name.startswith("delete"):
            ok = (home / ".zshrc.inferoid-test").exists()
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {'blocked' if ok else 'FILE WAS DELETED'}")
        else:
            worked = rc == 0
            ok = worked == should_work
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {'worked' if worked else 'blocked'}")
        failures += 0 if ok else 1
    for leftover in (home / "inferoid-escaped", home / "inferoid-escaped2", home / ".zshrc.inferoid-test", canary):
        try:
            leftover.unlink()
        except OSError:
            pass
    print("RESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(selftest(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inferoid-sandbox-test")))
