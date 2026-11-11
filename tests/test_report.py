import json
import os
import shutil
import subprocess
import threading
import venv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from inferoid.analysis import (
    classify,
    classify_heuristic,
    classify_step,
    local_chat,
    validate_local_endpoint,
)
from inferoid.parsers import scan, tool_fact
from inferoid.reporting import replay, report
from inferoid.storage import connect, estimated_cost, price_for, tier_for_ram, tier_spec


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def test_scan_parses_synthetic_claude_and_codex(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shadow_home = tmp_path / "shadow"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(shadow_home))
    timestamp = "2099-01-01T12:00:00Z"
    claude = home / ".claude/projects/demo/session.jsonl"
    write_jsonl(claude, [
        {"type": "user", "timestamp": timestamp, "cwd": "/private/project", "message": {"role": "user", "content": [{"type": "text", "text": "Explain this function"}]}},
        {"type": "assistant", "timestamp": timestamp, "isSidechain": False, "message": {"role": "assistant", "model": "claude-sonnet", "usage": {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7}, "content": [{"type": "text", "text": "A synthetic explanation."}, {"type": "tool_use", "name": "Read", "id": "t1"}]}},
        {"type": "future_unknown", "timestamp": timestamp},
        "not json",
    ])
    codex = home / ".codex/sessions/2099/01/01/rollout-test.jsonl"
    write_jsonl(codex, [
        {"type": "session_meta", "timestamp": timestamp, "payload": {"id": "codex-session", "cwd": "/private/project"}},
        {"type": "turn_context", "timestamp": timestamp, "payload": {"turn_id": "turn-1", "model": "gpt-5", "cwd": "/private/project"}},
        {"type": "response_item", "timestamp": timestamp, "payload": {"turn_id": "turn-1", "item": {"type": "UserMessage", "content": [{"type": "input_text", "text": "Run the tests"}]}}},
        {"type": "response_item", "timestamp": timestamp, "payload": {"turn_id": "turn-1", "item": {"type": "CommandExecution"}}},
        {"type": "response_item", "timestamp": timestamp, "payload": {"turn_id": "turn-1", "item": {"type": "AgentMessage", "content": [{"type": "output_text", "text": "Tests passed."}]}}},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {"turn_id": "turn-1", "usage": {"input_tokens": 200, "output_tokens": 80, "cached_input_tokens": 10, "cache_write_input_tokens": 2}}},
        {"type": "new_line_type", "timestamp": timestamp},
    ])

    stats = scan(30_000)
    conn = connect()
    turns = conn.execute("SELECT * FROM user_turns ORDER BY source").fetchall()
    calls = conn.execute("SELECT * FROM model_calls ORDER BY source").fetchall()
    assert len(turns) == 2
    assert len(calls) == 2
    assert all(len(row["project_hash"]) == 20 for row in turns)
    assert all("private/project" not in row["project_hash"] for row in turns)
    assert calls[0]["cache_read_tokens"] in {5, 10}
    assert stats["claude"]["unknown"] == 2
    assert stats["codex"]["unknown"] == 1

    counts = classify()
    assert counts["explain/question"] == 1
    assert counts["tests"] == 1


def test_codex_filechange_completion_counts_wrapper_patch_lines(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    write_jsonl(home / ".codex/sessions/2099/01/01/rollout-edit.jsonl", [
        {"type": "session_meta", "timestamp": timestamp, "payload": {"id": "s", "cwd": "/p"}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["Update the file"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]},
        }},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "custom_tool_call", "name": "exec",
            "input": 'const patch = "*** Begin Patch\n*** Update File: x.py\n-old\n+new\n*** End Patch"; await tools.apply_patch({patch});',
        }},
        {"type": "event_msg", "timestamp": timestamp, "payload": {"type": "item_completed", "item": {
            "type": "FileChange", "changes": {"x.py": {"type": "update", "unified_diff": "@@\n-old\n+new"}},
        }}},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {"usage": {"input_tokens": 10, "output_tokens": 2}}},
    ])
    scan(30_000)
    classify()
    call = connect().execute("SELECT * FROM model_calls").fetchone()
    facts = json.loads(call["tool_facts"])
    assert facts == [{"name": "FileChange", "added_lines": 1, "removed_lines": 1, "file_count": 1}]
    assert "Begin Patch" not in call["tool_facts"]
    assert call["step_class"] == "small edit"


def test_non_local_step_breakdown_classes():
    base = {"final_text_chars": 0, "response_text": "", "request_index": 0, "is_last": 1}
    assert classify_step({**base, "tool_facts": "[]", "tool_names": "[]"}) == "reasoning only"
    assert classify_step({**base, "tool_facts": '[{"name":"WebSearch"}]', "tool_names": "[]"}) == "web/search tool"
    assert classify_step({**base, "tool_facts": '[{"name":"Task"}]', "tool_names": "[]"}) == "subagent/task dispatch"
    assert classify_step({**base, "tool_facts": '[{"name":"MysteryTool"}]', "tool_names": "[]"}) == "unknown tool"
    assert classify_step({**base, "is_last": 0, "final_text_chars": 10, "response_text": "short", "tool_facts": "[]", "tool_names": "[]"}) == "intermediate text"


def test_memory_tier_for_24_gb_is_16_to_32():
    assert tier_for_ram(24) == "16"
    assert tier_spec("16") == {"ram_label": "16-32 GB", "max_context_tokens": 16_000}


def test_wheel_contains_data_and_runs_outside_repo(tmp_path):
    repo = Path(__file__).parents[1]
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    uv = shutil.which("uv")
    assert uv
    build_env = os.environ.copy()
    build_env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(repo)],
        check=True,
        capture_output=True,
        text=True,
        env=build_env,
    )
    wheel = next(wheel_dir.glob("inferoid-*.whl"))
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(venv_dir)
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [uv, "pip", "install", "--python", str(venv_python), "--no-index", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        env=build_env,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    machine = subprocess.run(
        [str(venv_python), "-m", "inferoid", "machine"],
        cwd=outside,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    machine_data = json.loads(machine.stdout)
    assert machine_data["memory_table_date"] == "2026-11-10"
    assert "tier_name" in machine_data
    price = subprocess.run(
        [str(venv_python), "-c", "from inferoid.storage import price_for; assert price_for('claude-opus-5-5')['output_per_million'] == 20.0"],
        cwd=outside,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert price.returncode == 0


def test_claude_context_and_cost_keep_disjoint_cache_tokens(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    write_jsonl(home / ".claude/projects/demo/session.jsonl", [
        {"type": "user", "timestamp": timestamp, "message": {"role": "user", "content": "Explain this"}},
        {"type": "assistant", "timestamp": timestamp, "message": {
            "role": "assistant", "model": "claude-opus-5-5",
            "usage": {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7},
            "content": [{"type": "text", "text": "Synthetic answer"}],
        }},
    ])
    scan(30_000)
    call = connect().execute("SELECT * FROM model_calls").fetchone()
    assert call["context_tokens"] == 112
    assert estimated_cost(call) == pytest.approx(0.001236)


def test_codex_context_and_cost_use_per_call_usage_not_turn_totals(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    write_jsonl(home / ".codex/sessions/2099/01/01/rollout-cost.jsonl", [
        {"type": "session_meta", "timestamp": timestamp, "payload": {"id": "s", "cwd": "/p"}},
        {"type": "turn_context", "timestamp": timestamp, "payload": {"model": "gpt-5.6-luna"}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["Calculate this"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]},
        }},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {
            "usage": {"input_tokens": 14_453, "cached_input_tokens": 10_240, "cache_write_input_tokens": 7, "output_tokens": 219},
            "turn_token_usage": {"input_tokens": 99_999, "output_tokens": 88_888},
        }},
    ])
    scan(30_000)
    call = connect().execute("SELECT * FROM model_calls").fetchone()
    assert call["input_tokens"] == 14_453
    assert call["cache_read_tokens"] == 10_240
    assert call["context_tokens"] == 14_460
    assert estimated_cost(call) == pytest.approx(0.0013102)


def test_codex_command_completion_replaces_js_wrapper_with_argv(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    write_jsonl(home / ".codex/sessions/2099/01/01/rollout-shell.jsonl", [
        {"type": "session_meta", "timestamp": timestamp, "payload": {"id": "s", "cwd": "/p"}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["Inspect status"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]},
        }},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "custom_tool_call", "call_id": "exec-1", "name": "exec",
            "input": 'const result = await tools.exec_command({cmd: "const wrapper = true"});',
        }},
        {"type": "event_msg", "timestamp": timestamp, "payload": {
            "type": "item_completed", "item": {"type": "CommandExecution", "id": "exec-1",
                "command": ["/bin/bash", "-lc", "cd /p && git status && sed -n '1,2p' file"]},
        }},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }},
    ])
    scan(30_000)
    call = connect().execute("SELECT * FROM model_calls").fetchone()
    fact = json.loads(call["tool_facts"])[0]
    assert fact["command_first_words"] == ["git status", "sed -n"]
    assert "const" not in fact["command_first_words"]
    assert classify_step(call) == "read-only shell dispatch"


@pytest.mark.parametrize(("command", "expected"), [
    ("cd dir && git status", "read-only shell dispatch"),
    ("FOO=bar timeout 5 rg needle | sed -n '1,2p'", "read-only shell dispatch"),
    ("git status && git commit -am update", "mutating shell dispatch"),
    ("sed -i s/old/new/ file", "mutating shell dispatch"),
    ("sed -n -i s/old/new/ file", "mutating shell dispatch"),
    ("echo result > output.txt", "mutating shell dispatch"),
    ("find . -type f | tee files.txt", "mutating shell dispatch"),
])
def test_shell_read_only_requires_every_segment(command, expected):
    fact = tool_fact("Bash", {"command": command})
    call = {"tool_facts": json.dumps([fact]), "final_text_chars": 0, "response_text": "", "request_index": 0, "is_last": 1}
    assert classify_step(call) == expected


def test_export_shape_has_user_and_no_response_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    conn = connect()
    conn.execute("INSERT INTO user_turns(id,user,source,session_id,project_hash,timestamp,text,turn_index) VALUES(?,?,?,?,?,?,?,?)", ("t", "local", "claude", "s", "hash", "2099-01-01", "synthetic prompt", 1))
    conn.execute("INSERT INTO model_calls(id,user_turn_id,source,session_id,project_hash,timestamp,model,response_text) VALUES(?,?,?,?,?,?,?,?)", ("c", "t", "claude", "s", "hash", "2099-01-01", "model", "synthetic answer"))
    conn.commit()
    from inferoid.cli import export_json
    export_json()
    result = json.loads(capsys.readouterr().out)
    assert result["user"] == "local"
    assert result["turns"][0]["text"] == "synthetic prompt"
    assert "response_text" not in result["turns"][0]["calls"][0]


def test_codex_real_prompt_shape_starts_unique_turns_and_assigns_calls(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    codex = home / ".codex/sessions/2099/01/01/rollout-real.jsonl"
    write_jsonl(codex, [
        {"type": "session_meta", "timestamp": timestamp, "payload": {"id": "session", "cwd": "/p"}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["injected context"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["environments.environment_context"], "turn_id": "p1"},
        }},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["first real prompt"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"], "turn_id": "p1"},
        }},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {"turn_id": "p1", "usage": {"input_tokens": 10, "output_tokens": 2}}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["second real prompt"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"], "turn_id": "p1"},
        }},
        {"type": "token_usage_record", "timestamp": timestamp, "payload": {"turn_id": "p1", "usage": {"input_tokens": 20, "output_tokens": 4}}},
        {"type": "response_item", "timestamp": timestamp, "payload": {
            "type": "message", "role": "user", "content": ["third real prompt without a model call"],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"], "turn_id": "p1"},
        }},
    ])
    scan(30_000)
    conn = connect()
    turns = conn.execute("SELECT * FROM user_turns ORDER BY turn_index").fetchall()
    calls = conn.execute("SELECT * FROM model_calls ORDER BY timestamp,id").fetchall()
    assert [turn["text"] for turn in turns] == ["first real prompt", "second real prompt", "third real prompt without a model call"]
    assert len({turn["id"] for turn in turns}) == 3
    assert [call["user_turn_id"] for call in calls] == [turn["id"] for turn in turns[:2]]


def test_claude_meta_and_tool_result_records_do_not_start_turn(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    claude = home / ".claude/projects/demo/session.jsonl"
    write_jsonl(claude, [
        {"type": "user", "timestamp": timestamp, "isMeta": True, "message": {"role": "user", "content": [{"type": "text", "text": "hook output"}]}},
        {"type": "user", "timestamp": timestamp, "message": {"role": "user", "content": [{"type": "tool_result", "content": "command output"}]}},
        {"type": "user", "timestamp": timestamp, "message": {"role": "user", "content": [{"type": "text", "text": "human request"}]}},
        {"type": "assistant", "timestamp": timestamp, "message": {"role": "assistant", "model": "claude", "usage": {}, "content": [{"type": "text", "text": "answer"}]}},
    ])
    scan(30_000)
    conn = connect()
    turns = conn.execute("SELECT text FROM user_turns").fetchall()
    assert [turn["text"] for turn in turns] == ["human request"]


def test_claude_headless_string_prompt_classifies_and_flags_cache_backend(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    target = home / ".claude/projects/synthetic/session.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(
        (Path(__file__).parent / "fixtures/claude-headless-ollama.jsonl").read_text(),
        encoding="utf-8",
    )

    stats = scan(30_000)
    assert stats["claude"]["turns"] == 1
    assert stats["claude"]["calls"] == 4
    assert stats["claude"]["cache_context_warning"] is True

    counts = classify()
    assert counts["tests"] == 1
    turn = connect().execute("SELECT text FROM user_turns").fetchone()
    assert turn["text"] == "Run the test suite."

    monkeypatch.setattr(
        "inferoid.reporting.detect_machine",
        lambda: {"os": "Linux", "os_version": "test", "chip": "chip", "ram_gb": 16, "ollama_models": []},
    )
    _, _, summary = report()
    assert summary["local_model_calls"] == 4
    assert summary["unknown_model_calls"] == 0
    assert summary["context_warning"] == (
        "Context accounting warning: context may be under-reported for this backend because repeated "
        "assistant blocks for one request disagree about cache-read tokens."
    )
    assert "context may be under-reported for this backend" in (tmp_path / "shadow/report.html").read_text()


def test_empty_text_is_unclassified_and_not_local_capable():
    turn = {"text": ""}
    category, size, _ = classify_heuristic(turn, [])
    assert category == "unclassified"
    assert size == "small"


def test_docs_classifier_avoids_product_and_folder_phrases():
    for text in ("Open Google Docs", "Search the docs folder", "Read the documentation"): 
        category, _, _ = classify_heuristic({"text": text}, [])
        if text == "Read the documentation":
            assert category == "docs/commit message"
        else:
            assert category != "docs/commit message"


def test_endpoint_must_be_localhost():
    assert validate_local_endpoint("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"
    assert validate_local_endpoint("http://[::1]:8080/v1") == "http://[::1]:8080/v1"
    with pytest.raises(ValueError):
        validate_local_endpoint("http://example.com/v1")
    with pytest.raises(ValueError):
        classify(endpoint="http://example.com/v1")


def test_report_gates_local_share_by_size_and_reports_unknown_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    monkeypatch.setattr("inferoid.reporting.detect_machine", lambda: {"os": "Linux", "os_version": "test", "chip": "chip", "ram_gb": 64, "recommended_class": "test", "ollama_models": []})
    conn = connect()
    turns = [
        ("small", "small edit", "small"),
        ("large", "small edit", "large"),
        ("unknown", "unclassified", "small"),
    ]
    for turn_id, task_class, size in turns:
        conn.execute("INSERT INTO user_turns(id,user,source,session_id,project_hash,timestamp,text,turn_index,task_class,size) VALUES(?,?,?,?,?,?,?,?,?,?)", (turn_id, "local", "codex", "s", "hash", "2099-01-01", "prompt", 1, task_class, size))
    for call_id, turn_id, model in (("c1", "small", "claude-opus-5-5"), ("c2", "large", "claude-opus-5-5"), ("c3", "unknown", "codex-auto-review")):
        conn.execute("INSERT INTO model_calls(id,user_turn_id,source,session_id,project_hash,timestamp,model,input_tokens,output_tokens) VALUES(?,?,?,?,?,?,?,?,?)", (call_id, turn_id, "codex", "s", "hash", "2099-01-01", model, 100, 100))
    conn.commit()
    _, share, summary = report()
    assert summary["local_capable_calls"] == 1
    assert summary["classified_calls"] == 2
    assert summary["unclassified_calls"] == 1
    assert summary["unknown_model_calls"] == 1
    assert summary["headline_share"] == 50.0
    assert share.startswith("50.0% of my Claude Code + Codex calls")


def test_price_matching_is_specific_and_unknown_is_not_default():
    assert price_for("claude-opus-5-5")["output_per_million"] == 20.0
    assert price_for("codex-auto-review") == {}
    assert price_for("gpt-oss:20b")["label"] == "local model ($0)"


def test_openai_compatible_local_chat_returns_ollama_durations(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "ok"}}],
                "prompt_eval_duration": 1_000_000,
                "eval_duration": 2_000_000,
            }).encode()

    monkeypatch.setattr("inferoid.analysis.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    text, metrics = local_chat("local", "prompt", endpoint="http://127.0.0.1:8080/v1")
    assert text == "ok"
    assert metrics["wall_time_ms"] == 3.0


def test_replay_persists_task_wall_time(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    conn = connect()
    conn.execute("INSERT INTO user_turns(id,user,source,session_id,project_hash,timestamp,text,turn_index,task_class,size) VALUES(?,?,?,?,?,?,?,?,?,?)", ("t", "local", "codex", "s", "hash", "2099-01-01", "Explain this", 1, "explain/question", "small"))
    conn.execute("INSERT INTO model_calls(id,user_turn_id,source,session_id,project_hash,timestamp,model,response_text) VALUES(?,?,?,?,?,?,?,?)", ("c", "t", "codex", "s", "hash", "2099-01-01", "gpt-5.6-luna", "Recorded"))
    conn.commit()
    monkeypatch.setattr("inferoid.reporting.local_chat", lambda *args, **kwargs: ("Local answer", {"wall_time_ms": 12.5}))
    monkeypatch.setattr("inferoid.reporting.ollama_chat", lambda *args, **kwargs: '{"outcome":"as good","score":0.9,"reason":"ok"}')
    replay(1, "local")
    row = conn.execute("SELECT wall_time_ms FROM replay_results").fetchone()
    assert row["wall_time_ms"] == 12.5
    capsys.readouterr()


def test_per_call_features_positions_and_step_classes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    claude = home / ".claude/projects/demo/session.jsonl"
    write_jsonl(claude, [
        {"type": "user", "timestamp": timestamp, "cwd": "/private/project", "message": {"role": "user", "content": [{"type": "text", "text": "Inspect and update the file"}]}},
        {"type": "assistant", "timestamp": timestamp, "message": {"role": "assistant", "model": "claude-sonnet", "usage": {"input_tokens": 100, "output_tokens": 4, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 3}, "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "/private/project/a.py"}}]}},
        {"type": "assistant", "timestamp": timestamp, "message": {"role": "assistant", "model": "claude-sonnet", "usage": {"input_tokens": 200, "output_tokens": 8, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 5}, "content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "/private/project/a.py", "old_string": "one\ntwo", "new_string": "one\nthree"}}]}},
    ])
    scan(30_000)
    classify()
    conn = connect()
    calls = conn.execute("SELECT * FROM model_calls ORDER BY request_index").fetchall()
    assert [call["context_tokens"] for call in calls] == [123, 235]
    assert [(call["request_index"], call["is_first"], call["is_last"]) for call in calls] == [(0, 1, 0), (1, 0, 1)]
    assert json.loads(calls[0]["tool_facts"])[0]["key_arg_kind"] == "path"
    assert json.loads(calls[1]["tool_facts"])[0]["added_lines"] == 1
    assert [call["step_class"] for call in calls] == ["read/search dispatch", "small edit"]


def test_replay_uses_fake_local_http_server(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadow"))
    timestamp = "2099-01-01T12:00:00Z"
    claude = home / ".claude/projects/demo/session.jsonl"
    write_jsonl(claude, [
        {"type": "user", "timestamp": timestamp, "cwd": "/private/project", "message": {"role": "user", "content": [{"type": "text", "text": "Read a.py"}]}},
        {"type": "assistant", "timestamp": timestamp, "message": {"role": "assistant", "model": "claude-sonnet", "usage": {"input_tokens": 100, "output_tokens": 2}, "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "/private/project/a.py"}}]}},
    ])
    scan(30_000)
    classify()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = {"choices": [{"message": {"content": '{"tool_name":"Read","arguments":{"file_path":"/private/project/a.py"}}'}}], "prompt_eval_duration": 1_000_000, "eval_duration": 2_000_000}
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        replay(1, "fake", f"http://127.0.0.1:{server.server_port}/v1", "16")
    finally:
        server.shutdown()
        thread.join(timeout=2)
    row = connect().execute("SELECT outcome,agreement,wall_time_ms,agreement_basis FROM replay_results").fetchone()
    assert row["outcome"] == "same action"
    assert row["agreement"] == 1
    assert row["wall_time_ms"] == 3.0
    assert row["agreement_basis"] == "objective dispatch comparison"
    capsys.readouterr()
