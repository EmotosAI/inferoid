import io
import json
import subprocess
from pathlib import Path

from inferoid import claude, cleanproxy, dispatch, mcp, setup


def _repo(path: Path) -> None:
    path.mkdir()
    (path / "start.txt").write_text("start\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=test@localhost",
                    "commit", "-qm", "init"], check=True)


class _Proxy:
    url = "http://127.0.0.1:1"

    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        return self

    def stop(self):
        pass


def _fake_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(dispatch.bench, "machine_memory", lambda: {"ram_gb": 24.0, "wired_limit_gb": 0.0})
    monkeypatch.setattr(claude, "pick_model", lambda _memory: ("test-model", 1.0))
    monkeypatch.setattr(dispatch.bench, "_model_info", lambda _model: {"size_gb": 1.0})
    monkeypatch.setattr(claude, "memory_problem", lambda *_args: "")
    monkeypatch.setattr(claude, "context_model", lambda model, _cap: model)
    monkeypatch.setattr(claude, "ensure_loaded", lambda _model: "")
    monkeypatch.setattr(claude, "loaded_context", lambda _model: 0)
    monkeypatch.setattr(claude, "system_used_gb", lambda: 1.0)
    monkeypatch.setattr(claude, "_generate", lambda *_args: "")
    monkeypatch.setattr(cleanproxy, "CleanProxy", _Proxy)


def test_run_applies_only_after_check_and_keeps_failure_worktree(tmp_path, monkeypatch):
    _fake_engine(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    _repo(repo)

    def fake_run(*_args, **kwargs):
        Path(kwargs.get("worktree") or _args[2], "changed.txt").write_text("done\n")
        return {"result_seen": True, "turns": 1, "tool_calls": 2, "input_tokens": 3, "output_tokens": 4,
                "errors": [], "timed_out": False, "interrupted": False, "wall_s": 0.1}

    monkeypatch.setattr(dispatch.bench, "run_claude", fake_run)
    passed = dispatch.run_task(project=repo, task="make the change", check="test -f changed.txt")
    assert passed["status"] == "done" and passed["applied"] is True
    assert (repo / "changed.txt").read_text() == "done\n"

    failed = dispatch.run_local_task(repo, "make another change", check="false")
    failed = dispatch.finish_task(repo, failed)
    assert failed["status"] == "check_failed" and not failed["applied"]
    assert Path(failed["worktree"]).exists()
    dispatch.bench.remove_isolated_copy(repo, Path(failed["worktree"]), True)


def test_run_refuses_outside_git_and_json_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch.bench, "_git_repo", lambda _path: False)
    value = dispatch.run_task(project=tmp_path, task="do not run")
    assert value["status"] == "error"
    assert value["applied"] is False
    assert {"status", "applied", "files_changed", "diffstat", "check_output_tail", "wall_s", "tool_calls",
            "input_tokens", "output_tokens", "model", "notes"} <= set(value)


def test_mcp_round_trip_with_fake_runner():
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "local_agent",
            "arguments": {"task": "fix it", "check": "test -f done", "files": ["src/a.py"]}}},
        {"jsonrpc": "2.0", "id": 4, "method": "ping"},
    ]
    called = {}

    def fake_runner(**kwargs):
        called.update(kwargs)
        return {"status": "done", "applied": True}

    input_stream = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n")
    output_stream = io.StringIO()
    mcp.serve(input_stream, output_stream, runner=fake_runner)
    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[0]["result"]["serverInfo"]["name"] == "inferoid"
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {"local_agent", "local_status"}
    assert json.loads(responses[2]["result"]["content"][0]["text"])["status"] == "done"
    assert responses[3]["result"] == {}
    assert called["task"] == "fix it" and called["timeout"] == 900


def test_setup_codex_and_cursor_writers_are_backed_up_and_idempotent(tmp_path):
    codex = tmp_path / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text("[profiles]\nname = \"test\"\n")
    setup.configure_codex(tmp_path, out=lambda _line: None)
    setup.configure_codex(tmp_path, out=lambda _line: None)
    text = codex.read_text()
    assert codex.with_name("config.toml.inferoid-bak").exists()
    assert text.count("[mcp_servers.inferoid]") == 1
    assert 'args = ["mcp"]' in text

    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir()
    cursor.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}) + "\n")
    setup.configure_cursor(tmp_path, out=lambda _line: None)
    setup.configure_cursor(tmp_path, out=lambda _line: None)
    data = json.loads(cursor.read_text())
    assert cursor.with_name("mcp.json.inferoid-bak").exists()
    assert set(data["mcpServers"]) == {"other", "inferoid"}


def test_repo_config_links_are_not_diffed_back(tmp_path):
    import json as _json
    import subprocess as sp
    from inferoid import dispatch
    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n")
    sp.run(["git", "-C", str(repo), "add", "."], check=True)
    sp.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], check=True)
    (repo / ".inferoid.json").write_text(_json.dumps({"link": ["data"], "allow_bash": ["make"]}))
    assert dispatch.repo_config(repo)["link"] == ["data"]
    wt = tmp_path / "wt"
    sp.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(wt)], check=True)
    (tmp_path / "datasrc").mkdir()
    (wt / "data").symlink_to(tmp_path / "datasrc")
    (wt / "a.txt").write_text("b\n")
    diff, paths, _ = dispatch._patch_info(wt, ["data"])
    assert paths == ["a.txt"] and "data" not in diff
