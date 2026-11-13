"""Kernel-enforced sandbox for local workers: macOS (Seatbelt, via sandbox-exec), Linux and WSL2 (bubblewrap).

A supervisor reviews work after it runs; it cannot stop a command that already deleted or copied something. The local
model's Claude Code process (and every command it starts, including `python -c` and shell tricks) runs inside a
profile that:
- allows writes only in the task worktree, temp folders and Claude Code's own state,
- denies reads of secrets (SSH, cloud and GPG keys, Keychains, browser profiles, ~/.config, .env files),
- denies all network except localhost (the model talks only to Ollama or a local engine on this Mac).
Linux and Windows (WSL2): bubblewrap gives the same three rules. The network is cut entirely (a new network namespace);
the worker still reaches the local model because a small forwarder listens on the same localhost port inside the
sandbox and relays to the real engine through a Unix socket.
Other systems: callers print a warning and run unsandboxed only if the user allows it.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import uuid
from pathlib import Path

SECRET_DIRS = [".ssh", ".aws", ".gnupg", ".config", ".kube", ".docker", ".azure", ".netrc", ".npmrc", ".pypirc",
               ".git-credentials", "Library/Keychains", "Library/Cookies", "Library/Mail", "Library/Messages",
               "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
               "Library/Safari", "Library/Application Support/com.apple.TCC", ".inferoid/config.json",
               ".forge-tokens", ".codex", ".openrouter-key"]


def kind() -> str:
    """seatbelt (macOS), bubblewrap (Linux, WSL2) or "" when no sandbox is available."""
    if platform.system() == "Darwin":
        return "seatbelt" if shutil.which("sandbox-exec") else ""
    if platform.system() == "Linux" and shutil.which("bwrap"):
        try:  # user namespaces can be disabled; prove bwrap actually starts
            ok = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-net", "true"], capture_output=True, timeout=10, check=False).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        return "bubblewrap" if ok else ""
    return ""


def available() -> bool:
    return bool(kind())


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


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


def wrap(args: list[str], worktree: Path, extra_write: list[Path] | None = None,
         env: dict[str, str] | None = None) -> list[str]:
    """Wrap a worker command in this system's sandbox. On Linux, env (the worker's environment) names the local
    endpoints the worker must still reach (ANTHROPIC_BASE_URL, OLLAMA_HOST); those localhost ports are forwarded."""
    if platform.system() == "Linux":
        return wrap_linux(args, worktree, extra_write=extra_write, ports=local_ports(env or {}))
    return ["sandbox-exec", "-p", profile(worktree, extra_write=extra_write), *args]


def local_ports(env: dict[str, str]) -> list[int]:
    ports = {11434}
    for key in ("ANTHROPIC_BASE_URL", "OLLAMA_HOST", "OPENAI_BASE_URL"):
        url = env.get(key, "")
        if not url:
            continue
        u = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        if u.hostname in ("127.0.0.1", "localhost", "::1"):
            ports.add(u.port or (443 if u.scheme == "https" else 80))
    return sorted(ports)


def _relay(a: socket.socket, b: socket.socket) -> None:
    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (src, dst):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
    threading.Thread(target=pump, args=(a, b), daemon=True).start()
    threading.Thread(target=pump, args=(b, a), daemon=True).start()


def _host_bridge(port: int, path: str) -> None:
    """Host side: Unix socket at path -> 127.0.0.1:port. Lives as long as this process."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(64)

    def loop() -> None:
        while True:
            try:
                conn, _ = srv.accept()
                up = socket.create_connection(("127.0.0.1", port), timeout=30)
                up.settimeout(None)
                _relay(conn, up)
            except OSError:
                continue
    threading.Thread(target=loop, name=f"inferoid-sandbox-bridge-{port}", daemon=True).start()


def wrap_linux(args: list[str], worktree: Path, home: Path | None = None, extra_write: list[Path] | None = None,
               ports: list[int] | None = None) -> list[str]:
    home = (home or Path.home()).resolve()
    worktree = Path(worktree).resolve()
    bridge_dir = Path(tempfile.gettempdir()) / f"inferoid-sb-{uuid.uuid4().hex[:8]}"
    bridge_dir.mkdir(mode=0o700)
    forwards = []
    for port in ports or [11434]:
        path = str(bridge_dir / f"{port}.sock")
        _host_bridge(port, path)
        forwards += ["--forward", f"{port}={path}"]
    for d in (home / ".claude", home / ".cache"):
        d.mkdir(parents=True, exist_ok=True)
    b = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
         "--bind", "/tmp", "/tmp", "--bind", str(worktree), str(worktree),
         "--bind", str(home / ".claude"), str(home / ".claude"), "--bind", str(home / ".cache"), str(home / ".cache")]
    if (home / ".claude.json").exists():
        b += ["--bind", str(home / ".claude.json"), str(home / ".claude.json")]
    for extra in extra_write or []:
        b += ["--bind", str(Path(extra).resolve()), str(Path(extra).resolve())]
    for d in SECRET_DIRS:  # hide secrets: empty tmpfs over folders, /dev/null over files
        target = home / d
        if target.is_dir():
            b += ["--tmpfs", str(target)]
        elif target.exists():
            b += ["--ro-bind", "/dev/null", str(target)]
    for env_file in _env_files(worktree):
        b += ["--ro-bind", "/dev/null", str(env_file)]
    return [*b, "--", sys.executable, "-m", "inferoid.sandbox", "--inside", *forwards, "--", *args]


def _env_files(root: Path, limit: int = 2000) -> list[Path]:
    found, seen = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", ".venv", "venv", "__pycache__")]
        for name in filenames:
            seen += 1
            if name == ".env" or name.startswith(".env."):
                found.append(Path(dirpath) / name)
        if seen > limit * 50:
            break
    return found[:limit]


def _inside(argv: list[str]) -> int:
    """Runs inside the Linux sandbox: listen on each forwarded localhost port, relay to the host's Unix socket,
    then run the worker command and return its exit code."""
    forwards, i = [], 0
    while i < len(argv) and argv[i] != "--":
        if argv[i] == "--forward":
            port, path = argv[i + 1].split("=", 1)
            forwards.append((int(port), path))
            i += 2
        else:
            i += 1
    cmd = argv[i + 1:]
    subprocess.run(["ip", "link", "set", "lo", "up"], capture_output=True, check=False)  # bwrap usually does this already
    for port, path in forwards:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(64)

        def loop(srv: socket.socket = srv, path: str = path) -> None:
            while True:
                try:
                    conn, _ = srv.accept()
                    up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    up.connect(path)
                    _relay(conn, up)
                except OSError:
                    continue
        threading.Thread(target=loop, daemon=True).start()
    return subprocess.call(cmd)


def selftest(workdir: Path) -> int:
    """Run escape attempts inside the profile. Every attempt must fail; work inside the folder must succeed."""
    if not available():
        print("no sandbox on this system: macOS needs sandbox-exec; Linux and WSL2 need bubblewrap (sudo apt install bubblewrap)")
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
        rc = subprocess.run(wrap(cmd, workdir), capture_output=True, text=True, timeout=30, check=False).returncode
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
    if len(sys.argv) > 1 and sys.argv[1] == "--inside":
        raise SystemExit(_inside(sys.argv[2:]))
    raise SystemExit(selftest(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inferoid-sandbox-test")))
