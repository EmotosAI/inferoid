import pytest
from pathlib import Path

from inferoid import sandbox


def test_profile_limits_writes_secrets_and_network(tmp_path):
    prof = sandbox.profile(tmp_path / "wt", home=tmp_path / "home")
    assert "(deny network*)" in prof and 'remote ip "localhost:*"' in prof
    assert 'local ip' not in prof  # combined with the unix-socket rule it let internet traffic through
    assert "(deny file-write*)" in prof and str((tmp_path / "wt").resolve()) in prof
    assert str(tmp_path / "home" / ".ssh") in prof and str(tmp_path / "home" / ".aws") in prof


def test_wrap_prefixes_sandbox_exec(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Darwin")
    args = sandbox.wrap(["claude", "-p", "x"], tmp_path)
    assert args[:2] == ["sandbox-exec", "-p"] and args[-3:] == ["claude", "-p", "x"]


def test_local_ports_from_worker_env():
    env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:43210", "OPENAI_BASE_URL": "https://api.example.com/v1",
           "OLLAMA_HOST": "localhost:11500"}
    assert sandbox.local_ports(env) == [11434, 11500, 43210]


def test_linux_wrap_cuts_network_hides_secrets_and_env_files(tmp_path):
    home, wt = tmp_path / "home", tmp_path / "wt"
    (home / ".ssh").mkdir(parents=True)
    (home / ".netrc").write_text("x")
    (wt / "sub").mkdir(parents=True)
    (wt / ".env").write_text("SECRET=1")
    (wt / "sub" / ".env.local").write_text("SECRET=2")
    args = sandbox.wrap_linux(["claude", "-p", "x"], wt, home=home, ports=[11434])
    joined = " ".join(args)
    assert args[0] == "bwrap" and "--unshare-net" in args and "--ro-bind / /" in joined
    assert f"--tmpfs {home / '.ssh'}" in joined and f"--ro-bind /dev/null {home / '.netrc'}" in joined
    assert f"--ro-bind /dev/null {(wt / '.env').resolve()}" in joined
    assert f"--ro-bind /dev/null {(wt / 'sub' / '.env.local').resolve()}" in joined
    assert f"--bind {wt.resolve()} {wt.resolve()}" in joined and "--forward" in args
    assert args[-3:] == ["claude", "-p", "x"]


@pytest.mark.skipif(sandbox.kind() != "bubblewrap", reason="needs bubblewrap with user namespaces")
def test_linux_sandbox_blocks_escapes_and_keeps_localhost(tmp_path):
    import http.server, socket, subprocess, sys, threading
    wt = tmp_path / "wt"
    wt.mkdir()
    srv = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from pathlib import Path as _P
    outside = _P.home() / ".inferoid-sandbox-test-outside"  # /tmp is writable by design; home is not
    outside.unlink(missing_ok=True)
    code = (f"import urllib.request, pathlib\n"
            f"pathlib.Path({str(wt / 'ok.txt')!r}).write_text('in')\n"
            f"try:\n    pathlib.Path({str(outside)!r}).write_text('out'); print('WROTE-OUTSIDE')\nexcept OSError: print('outside-blocked')\n"
            f"print(urllib.request.urlopen('http://127.0.0.1:{port}/', timeout=5).status)\n"
            f"try:\n    urllib.request.urlopen('https://example.com', timeout=5); print('INTERNET')\nexcept Exception: print('internet-blocked')\n")
    args = sandbox.wrap_linux([sys.executable, "-c", code], wt, home=tmp_path / "home", ports=[port])
    out = subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    srv.shutdown()
    leaked = outside.exists()
    outside.unlink(missing_ok=True)
    assert (wt / "ok.txt").read_text() == "in" and not leaked
    assert "outside-blocked" in out and "200" in out and "internet-blocked" in out
