import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from inferoid import bench, claude, forge


def test_allowlist_drops_private_fields():
    report = {
        "schema": 1, "inferoid_version": "0.1.4", "submitted_at": "now",
        "prompt": "private", "output": "private", "path": "/private", "username": "me",
        "machine": {"chip": "x", "ram_gb": 24, "gpu_budget_gb": 16, "os": "Linux", "secret": "x"},
        "engine": {"name": "ollama", "version": "x", "secret": "x"},
        "model": {"name": "m", "size_gb": 1, "family": "m", "quant": "q", "output": "x"},
        "task": {"pack": "p", "id": "t", "kind": "coding", "brief": "private"},
        "settings": {"context_cap": 32768, "lean": True, "effort": "low", "timeout_s": 10},
        "result": {"status": "pass", "check_passed": True, "failure_kind": "none", "wall_s": 1,
                   "first_reply_s": 0.1, "turns": 1, "tool_calls": 2, "input_tokens": 3,
                   "output_tokens": 4, "peak_model_mem_gb": 5, "peak_swap_gb": 0,
                   "memory_pressure": "normal", "output": "private"},
    }
    clean = forge.allowlisted_report(report)
    encoded = json.dumps(clean)
    assert "private" not in encoded and "/private" not in encoded and "username" not in encoded
    assert set(clean) == {"schema", "inferoid_version", "submitted_at", "machine", "engine", "model", "task", "settings", "result"}


@pytest.mark.parametrize(
    ("state", "check", "changed", "expected"),
    [
        ({"timed_out": True}, False, True, "timeout"),
        ({"errors": ["prompt too long"], "result_seen": False}, False, False, "context_exhausted"),
        ({"errors": ["model requires 20 GiB"], "result_seen": False}, False, False, "out_of_memory"),
        ({"errors": ["invalid tool input"], "result_seen": True}, False, True, "tool_format"),
        ({"result_seen": False}, False, False, "no_output"),
        ({"result_seen": True}, False, True, "check_failed"),
    ],
)
def test_failure_classification(state, check, changed, expected):
    assert bench.classify_failure(state, check, changed, 32768) == expected


def test_memory_samples_use_ollama_and_macos_commands(monkeypatch):
    monkeypatch.setattr(bench, "_ollama_json", lambda path: {"models": [{"name": "m", "size_vram": 2 * 1024**3}]})
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    def fake_check_output(command, **_kwargs):
        if command[0] == "sysctl":
            return "vm.swapusage: total = 2.00G  used = 512.00M  free = 1.50G"
        return "System-wide memory free percentage: 20%\n"
    monkeypatch.setattr(bench.subprocess, "check_output", fake_check_output)
    assert bench.loaded_model_memory("m") == 2
    assert bench.swap_used_gb() == 0.5
    assert bench.memory_pressure() == "normal"


def test_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "child-done"
    code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(10)']); time.sleep(10)")
    state = bench.run_claude([sys.executable, "-c", code], dict(os.environ), tmp_path, "m", 0.15, 32768)
    assert state["timed_out"] is True
    assert state["returncode"] is not None
    assert not marker.exists()


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "README.txt").write_text("original\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=test@localhost", "commit", "-qm", "init"], check=True)


def test_run_uses_and_removes_worktree_without_touching_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    brief = tmp_path / "brief.md"
    brief.write_text("Do the task.")
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "inferoid"))
    monkeypatch.setattr(bench, "machine_memory", lambda: {"ram_gb": 24.0, "wired_limit_gb": 0.0})
    monkeypatch.setattr(claude, "pick_model", lambda memory: ("m", 1.0))
    monkeypatch.setattr(claude, "memory_problem", lambda *args: "")
    monkeypatch.setattr(claude, "ensure_loaded", lambda model: "")
    monkeypatch.setattr(bench, "_engine_version", lambda: "test")

    def fake_run(args, env, worktree, model, timeout, context, transcript=None, baseline=0.0, limit=0.0):
        (worktree / "done.txt").write_text("done\n")
        return {"first_reply_s": 0.1, "turns": 1, "tool_calls": 1, "input_tokens": 2, "output_tokens": 3,
                "result_seen": True, "errors": [], "timed_out": False, "returncode": 0, "wall_s": 0.2,
                "peak_model_mem_gb": 1.0, "peak_swap_gb": 0.0, "memory_pressure": "normal"}

    monkeypatch.setattr(bench, "run_claude", fake_run)
    code, report_path = bench.run(repo, brief, "test -f done.txt", "pack", "task", no_submit=True)
    assert code == 0 and report_path.exists()
    assert not (repo / "done.txt").exists()
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout == ""
    report = json.loads(report_path.read_text())
    assert report["result"]["status"] == "pass"
    assert "done.txt" not in json.dumps(report)
    assert not list((tmp_path / "inferoid" / "bench" / "wt").glob("*"))


class _ForgeHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []
    status_code = 201

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.requests.append((self.path, self.headers.get("Authorization"), json.loads(body)))
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"id": "r1", "verdict": {"kind": "good"}}).encode())

    def log_message(self, *_args):
        return


def _server():
    _ForgeHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ForgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _report():
    return {"schema": 1, "inferoid_version": "0.1.4", "submitted_at": "now",
            "machine": {"chip": "x", "ram_gb": 1, "gpu_budget_gb": 1, "os": "x"},
            "engine": {"name": "ollama", "version": "x"}, "model": {"name": "m", "size_gb": 1, "family": "m", "quant": "q"},
            "task": {"pack": "p", "id": "t", "kind": "k"}, "settings": {"context_cap": 1, "lean": True, "effort": "low", "timeout_s": 1},
            "result": {"status": "pass", "check_passed": True, "failure_kind": "none", "wall_s": 1, "first_reply_s": 1,
                       "turns": 1, "tool_calls": 1, "input_tokens": 1, "output_tokens": 1, "peak_model_mem_gb": 1,
                       "peak_swap_gb": 0, "memory_pressure": "normal"}}


def test_submit_bearer_201_401_and_network(tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "home"))
    server, url = _server()
    try:
        ok, _, verdict = forge.submit(_report(), token="secret", url=url)
        assert ok and verdict == {"kind": "good"}
        assert _ForgeHandler.requests[0][1] == "Bearer secret"
        assert "prompt" not in _ForgeHandler.requests[0][2]
        _ForgeHandler.status_code = 401
        ok, message, _ = forge.submit(_report(), token="secret", url=url)
        assert not ok and "401" in message
    finally:
        server.shutdown()
    ok, message, _ = forge.submit(_report(), token="secret", url="http://127.0.0.1:1")
    assert not ok and "failed" in message


def test_login_permissions_and_status_hides_token(tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(forge, "_health", lambda url: "ok")
    forge.login("secret-token", "http://127.0.0.1:1234", share=False)
    path = forge.config_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = forge.status()
    assert "set" in lines[1] and "secret-token" not in "\n".join(lines)


def test_share_links_untracked_data_into_the_copy(tmp_path, monkeypatch):
    import subprocess

    from inferoid import bench

    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    proj.mkdir()
    subprocess.run(["git", "init", "-q", str(proj)], check=True)
    (proj / "a.txt").write_text("tracked")
    (proj / ".gitignore").write_text("analysis/\n")
    subprocess.run(["git", "-C", str(proj), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(proj), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], check=True)
    (proj / "analysis").mkdir()
    (proj / "analysis" / "grid.json").write_text("{}")
    (proj / "a.txt").write_text("changed, not committed")
    assert bench.dirty_paths(proj) == ["a.txt"]
    target, is_git = bench.make_isolated_copy(proj, "r1", ["analysis", "missing"], from_commit=True)
    assert is_git and (target / "analysis").is_symlink() and (target / "analysis" / "grid.json").exists()
    assert (target / "a.txt").read_text() == "tracked"
    bench.remove_isolated_copy(proj, target, is_git)
    # Default: the working tree (uncommitted changes and untracked, non-ignored files) is benchmarked.
    (proj / "new.py").write_text("x = 1\n")
    target2, _ = bench.make_isolated_copy(proj, "r2")
    assert (target2 / "a.txt").read_text() == "changed, not committed" and (target2 / "new.py").exists()
    assert not (target2 / "analysis").exists()  # ignored and not shared
    bench.remove_isolated_copy(proj, target2, True)
    assert not target.exists() and (proj / "analysis" / "grid.json").exists()



def test_no_output_when_agent_did_nothing():
    from inferoid import bench

    state = {"result_seen": True, "tool_calls": 0, "errors": [], "input_tokens": 900}
    assert bench.classify_failure(state, check_passed=False, changed=False, context=32768) == "no_output"
    state["tool_calls"] = 3
    assert bench.classify_failure(state, check_passed=False, changed=True, context=32768) == "check_failed"


def test_context_model_copies_chat_format(monkeypatch):
    from inferoid import claude

    created = []
    tags = {"models": [{"name": "gemma4-26b-reap25:latest"}]}
    monkeypatch.setattr(claude, "_ollama", lambda path: tags)
    base = "FROM /blobs/sha256-abc\nTEMPLATE {{ .Prompt }}\nRENDERER gemma4\nPARSER gemma4\nPARAMETER num_ctx 4096\n"

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = base if cmd[1] == "show" else ""
        if cmd[1] == "create":
            created.append(open(cmd[-1]).read())
        return R()

    name = claude.context_model("gemma4-26b-reap25", 131072, run=fake_run)
    assert name == "gemma4-26b-reap25-ctx128k"
    text = created[0]
    assert "FROM gemma4-26b-reap25\n" in text and "/blobs/" not in text
    assert "RENDERER gemma4" in text and "PARSER gemma4" in text
    assert "num_ctx 4096" not in text and "PARAMETER num_ctx 131072" in text


def test_local_errors_stay_local():
    from inferoid import forge

    report = {"schema": 1, "inferoid_version": "x", "submitted_at": "t",
              "machine": {"chip": "c", "ram_gb": 24, "gpu_budget_gb": 16.1, "os": "o"},
              "engine": {"name": "ollama", "version": "v"}, "model": {"name": "m", "size_gb": 1.0, "family": "f", "quant": "q"},
              "task": {"pack": "p", "id": "i", "kind": "k"},
              "settings": {"context_cap": 1, "lean": True, "effort": "low", "timeout_s": 1},
              "result": {"status": "error", "check_passed": False, "failure_kind": "engine_error", "wall_s": 0.0,
                         "first_reply_s": None, "turns": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "peak_model_mem_gb": 0.0, "peak_swap_gb": 0.0, "memory_pressure": "normal"},
              "local_only": {"errors": ["/Users/someone/secret/path failed"]}}
    assert "local_only" not in forge.allowlisted_report(report)


def test_output_tokens_sum_across_messages_and_estimate_when_engine_reports_zero():
    state = {"first_reply_s": None, "turns": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
             "result_seen": False, "errors": []}
    block = {"type": "tool_use", "name": "Grep", "input": {"pattern": "plan" * 50}}

    def event(mid, out):
        return json.dumps({"type": "assistant", "message": {"id": mid, "content": [block],
                                                            "usage": {"input_tokens": 1000, "output_tokens": out}}})

    bench._consume_stream(event("a", 120), state, 0.0)
    bench._consume_stream(event("b", 80), state, 0.0)
    assert state["output_tokens"] == 200  # summed, not the largest single message
    zero = {"first_reply_s": None, "turns": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "result_seen": False, "errors": []}
    bench._consume_stream(event("c", 0), zero, 0.0)
    bench._consume_stream(event("d", 0), zero, 0.0)
    assert zero["output_tokens"] > 100  # Ollama said 0 for 2 tool calls; estimated from the content instead


def test_ctrl_c_stops_claude_and_still_returns_state(monkeypatch, tmp_path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(bench.Monitor, "start", lambda self: None)
    monkeypatch.setattr(bench.Monitor, "finish", lambda self: None)
    calls = {"n": 0}
    real_get = bench.queue.Queue.get

    def interrupting_get(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return real_get(self, *a, **kw)

    monkeypatch.setattr(bench.queue.Queue, "get", interrupting_get)
    state = bench.run_claude([str(script)], dict(os.environ), tmp_path, "m", 60, 32768)
    assert state["interrupted"] and "Ctrl+C" in " ".join(state["errors"])


def test_runaway_reply_is_stopped(monkeypatch, tmp_path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(bench, "STALL_S", 1)
    monkeypatch.setattr(bench.Monitor, "start", lambda self: None)
    monkeypatch.setattr(bench.Monitor, "finish", lambda self: None)
    state = bench.run_claude([str(script)], dict(os.environ), tmp_path, "m", 60, 32768)
    assert state["stalled"] and not state["timed_out"]
    assert bench.classify_failure(state, False, False, 32768) == "no_output"


def test_memory_danger_stops_the_run(monkeypatch, tmp_path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(bench.Monitor, "start", lambda self: setattr(self, "danger", "swap grew by 3.2 GB"))
    monkeypatch.setattr(bench.Monitor, "finish", lambda self: None)
    state = bench.run_claude([str(script)], dict(os.environ), tmp_path, "m", 60, 32768)
    assert state["memory_stop"]
    assert bench.classify_failure(state, False, False, 32768) == "out_of_memory"


def _monitor(monkeypatch, swaps, model_gb=11.0, pressure="normal", wired=None):
    seq = iter(swaps)
    wired_seq = iter(wired or [5.0] * 20)
    monkeypatch.setattr(bench.claude, "system_wired_gb", lambda: next(wired_seq))
    monkeypatch.setattr(bench, "swap_used_gb", lambda: next(seq))
    monkeypatch.setattr(bench, "loaded_model_memory", lambda m: model_gb)
    monkeypatch.setattr(bench, "memory_pressure", lambda: pressure)
    return bench.Monitor("m")


def test_monitor_judges_swap_growth_not_leftover(monkeypatch):
    m = _monitor(monkeypatch, [2.15, 2.15, 2.18])  # start, then samples: 2.15 GB was there before the run
    m.sample(); m.sample()
    assert m.peak_swap < 0.1 and m.pressure == "normal" and not m.danger


def test_monitor_stops_on_1gb_swap_growth_and_stalls(monkeypatch):
    m = _monitor(monkeypatch, [2.0, 3.3])
    m.sample()
    assert "swap grew" in m.danger and m.pressure == "critical"
    m = _monitor(monkeypatch, [2.0, 2.0, 2.0])
    m.sample()
    m.last_sample -= 30  # 30 s with no sample = macOS stalled
    m.sample()
    assert "stalled" in m.danger


def test_monitor_stops_over_budget_and_on_jumps(monkeypatch):
    m = _monitor(monkeypatch, [0.0, 0.0], model_gb=17.0)
    m.limit = 16.1
    m.sample()
    assert "over the 16.1 GB" in m.danger
    # Wired memory jumping after warm-up stops the run; during warm-up it does not.
    m = _monitor(monkeypatch, [0.0] * 8, wired=[5.0, 9.0, 9.0, 9.0, 12.0])
    for _ in range(4):
        m.sample()
    assert not m.danger
    m.sample()
    assert "jumped" in m.danger


def test_under_reported_model_uses_estimate_not_file_cache(monkeypatch):
    # Ollama says 1.3 GB for 11.5 GB of weights: no budget stop from a figure that includes the file cache.
    m = _monitor(monkeypatch, [2.36] * 5, model_gb=1.3)
    m.weights, m.estimate, m.limit = 11.5, 13.9, 16.1
    m.sample(); m.sample()
    assert not m.danger and m.peak_model == 13.9


def test_failed_submit_is_queued_and_resent(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "home", lambda: tmp_path)
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"schema": 1}))
    bench.queue_for_forge(report)
    assert (tmp_path / "bench" / "outbox" / "r.json").exists()
    monkeypatch.setattr(bench.forge, "read_config", lambda: {"share_reports": True, "forge_token": "t"})
    monkeypatch.setattr(bench.forge, "submit", lambda data: (True, "ok", None))
    bench.flush_outbox()
    assert not (tmp_path / "bench" / "outbox" / "r.json").exists()


def test_local_report_keeps_local_only_but_submit_strips_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "home", lambda: tmp_path)
    report = {"schema": 1, "model": {"name": "m"}, "task": {"id": "t"},
              "local_only": {"errors": ["boom"], "clean_changes": {"thinking_disabled": 3}}}
    path = bench.write_report(report, tmp_path / "r.json")
    saved = json.loads(path.read_text())
    assert saved["local_only"]["clean_changes"]["thinking_disabled"] == 3
    assert "local_only" not in forge.allowlisted_report(saved)
